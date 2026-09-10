"""
Model transformations for transfer-learning experiments.

Input will be a trained model, and output will be a new model
with the implemented strategies. Eg frozen layers etc
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Dict, Tuple, Optional

import torch
from e3nn import o3

from mace.data.atom_data_loader import AtomDataMetadata
from mace.modules.blocks import (
    DifferenceDecoder
)

from mace.modules.lora import (
    LoRADenseLinear,
    LoRAFCLayer,
    LoRAO3Linear,
    inject_lora,
)


model_layers = [
    "node_embedding",
    "perm_encoder",
    "perm_decoder",
    "interactions",
    "products",
    "readouts",
    "invariant_readouts",
]


def _copy_model(model: torch.nn.Module) -> torch.nn.Module:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    return deepcopy(model)


def _reset_module_parameters(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
        elif isinstance(child, o3.Linear):
            with torch.no_grad():
                child.weight.normal_()
                child.bias.zero_()


def _zero_module_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def _initialise_correction_head(head: torch.nn.Module) -> None:
    """
    Helper to initialise the correction head instead of just deep copying it
    Makes it such that the initial outputs are almost 0 so the model starts closer 
    to the actual values. Else the initial input will be almost double the errors
    """
    _reset_module_parameters(head)

    for nac_readout in head.nac_readouts:
        output_layer = (
            nac_readout.linear
            if hasattr(nac_readout, "linear")
            else nac_readout.linear_2
        )
        _zero_module_parameters(output_layer)

    for soc_readout in head.socs_readouts:
        output_layer = (
            soc_readout.linear
            if hasattr(soc_readout, "linear")
            else soc_readout.linear_2
        )
        _zero_module_parameters(output_layer)

@dataclass
class NaiveStrategy:
    """
    Most basic naive strategy for TL, just copy the original model
    and retrain from the parameters learned without changing any of
    its architecture.
    """
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return _copy_model(model)


@dataclass
class FreezeStrategy:
    """
    Basic freezing strategy for the model to freeze certain layers
    """

    frozen_layers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_layers, tuple):
            raise TypeError("frozen_layers must be a tuple of layer names.")

        invalid_layers = [
            layer for layer in self.frozen_layers if layer not in model_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"frozen_layers must contain only layers from {model_layers}."
            )

        if set(self.frozen_layers) == set(model_layers):
            raise ValueError("Every trainable model layer cannot be frozen.")

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        transfer_model = _copy_model(model)

        for name, parameter in transfer_model.named_parameters():
            for layer in self.frozen_layers:
                if name == layer or name.startswith(f"{layer}."):
                    parameter.requires_grad_(False)

        return transfer_model


@dataclass
class LoRAStrategy:
    """
    After applying this strategy, all layers are frozen by default 
    To add low-rank adaptation to certain layers, specify in lora_layers
    To let certain layers be fully trainable (like naive), specify in full_train_layers

    Rank: rank (size) of the low-rank adaptation
    Alpha: scaling factor for the LoRA layers

    Actual LoRA implementation is done in mace.modules.lora, this strategy just applies it to the model
    """

    rank: int = 4
    alpha: float = 1.0
    lora_layers: Tuple[str, ...] = ()
    full_train_layers: Tuple[str, ...] = ()

    def __post_init__(self) -> None: 
        # validate rank and alpha
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer greater than zero.")
        if self.rank <= 0:
            raise ValueError("rank must be greater than zero.")

        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a finite numeric value greater than zero.")
        if not math.isfinite(float(self.alpha)) or self.alpha <= 0:
            raise ValueError("alpha must be a finite numeric value greater than zero.")

        # validate that lora_layers and full_train_layers contain valid layer names
        # and don't overlap
        self._validate_layer_tuple("lora_layers", self.lora_layers)
        self._validate_layer_tuple("full_train_layers", self.full_train_layers)

        overlap = sorted(set(self.lora_layers) & set(self.full_train_layers))
        if overlap:
            raise ValueError(
                f"Layers cannot be both LoRA-adapted and fully trained: {overlap}"
            )
        if not self.lora_layers and not self.full_train_layers:
            raise ValueError("At least one layer must be LoRA-adapted or fully trained.")

    @staticmethod
    def _validate_layer_tuple(name: str, layers: Tuple[str, ...]) -> None:
        if not isinstance(layers, tuple) or not all(isinstance(layer, str) for layer in layers):
            raise TypeError(f"{name} must be a tuple of layer names.")
        if len(layers) != len(set(layers)):
            raise ValueError(f"{name} cannot contain duplicate layer names.")

        invalid_layers = [layer for layer in layers if layer not in model_layers]
        if invalid_layers:
            raise ValueError(f"{name} must contain only layers from {model_layers}.")

    @staticmethod
    def _selected_module(
        model: torch.nn.Module, layer_name: str
    ) -> torch.nn.Module:
        module = getattr(model, layer_name, None)
        if module is None:
            raise ValueError(
                f"Selected layer '{layer_name}' is not present on the supplied model."
            )
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                f"Selected layer '{layer_name}' must be a torch.nn.Module."
            )
        return module

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        transfer_model = _copy_model(model)
        selected_modules = {
            layer_name: self._selected_module(transfer_model, layer_name)
            for layer_name in (*self.lora_layers, *self.full_train_layers)
        }

        for parameter in transfer_model.parameters():
            parameter.requires_grad_(False) # freeze all layers by default

        lora_types = (LoRAO3Linear, LoRADenseLinear, LoRAFCLayer)
        for layer_name in self.lora_layers:
            module = selected_modules[layer_name]
            inject_lora(module, rank=self.rank, alpha=self.alpha) # inject LoRA adapters to layers that are selected for LoRA
            if not any(isinstance(child, lora_types) for child in module.modules()):
                raise ValueError(
                    f"Selected LoRA layer '{layer_name}' contains no compatible layers."
                )

        for layer_name in self.full_train_layers:
            for parameter in selected_modules[layer_name].parameters():
                parameter.requires_grad_(True) # unfreeze layers that are selected for full training

        trainable_parameters = [
            parameter for parameter in transfer_model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("LoRA strategy produced a model with no trainable parameters.")

        # stuff below just double checks that all layers are in the correct state after applying the strategy
        for layer_name in self.lora_layers:
            module = selected_modules[layer_name]
            for child in module.modules():
                if isinstance(child, lora_types):
                    if not all(parameter.requires_grad for parameter in child.lora_A.parameters()):
                        raise RuntimeError("LoRA adapter parameters must be trainable.")
                    if not all(parameter.requires_grad for parameter in child.lora_B.parameters()):
                        raise RuntimeError("LoRA adapter parameters must be trainable.")
                    if any(parameter.requires_grad for parameter in child.base.parameters()):
                        raise RuntimeError("LoRA base parameters must be frozen.")

        for layer_name in self.full_train_layers:
            if not all(
                parameter.requires_grad
                for parameter in selected_modules[layer_name].parameters()
            ):
                raise RuntimeError(
                    f"Parameters in fully trained layer '{layer_name}' must be trainable."
                )

        selected_layer_names = set(self.lora_layers) | set(self.full_train_layers)
        for layer_name in model_layers:
            if layer_name in selected_layer_names:
                continue
            module = getattr(transfer_model, layer_name, None)
            if isinstance(module, torch.nn.Module) and any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                raise RuntimeError(
                    f"Parameters in unselected layer '{layer_name}' must be frozen."
                )
class MultiHeadCorrectionStrategy:
    """
    Duplicate a trained autoencoder head for multi-head training.
    This trainer assumes a route with routed data rather than separated training
    LR here will all be taken relative to the Trainer's LR
    """

    metadata: AtomDataMetadata
    gnn_lr: float = 0.01
    head_multipliers: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.metadata.num_heads < 2:
            raise ValueError("metadata must contain at least 2 heads.")

        expected_shape = (self.metadata.num_heads, self.metadata.num_elements)
        if self.metadata.atomic_energies.shape != expected_shape:
            raise ValueError(
                f"metadata.atomic_energies must have shape {expected_shape}."
            )

        self.num_heads = self.metadata.num_heads

        # Build the LR head vector 
        if self.head_multipliers is None: 
            # By default first head is 0.01 LR, the rest are 1 
            self._head_lr_vector = [0.01] + [1] * (self.num_heads - 1)
        else:
            # First check that the lr multiplier keys are the same as those in metadata
            if set(self.head_multipliers) != set(self.metadata.head_to_index):
                raise ValueError("head_multipliers must have the same head keys as metadata head to index")

            # Then define the vector in head lr based on the head to index order
            self._head_lr_vector = [
                float(self.head_multipliers[head_name])
                for head_name in self.metadata.head_to_index
            ]

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        transfer_model = _copy_model(model)

        if len(transfer_model.autoencoder_heads) != 1:
            raise ValueError(
                "MultiHeadStrategy expects a model with one template head."
            )

        # First copy the head and add the correction blocks
        template_head = transfer_model.autoencoder_heads[0]
        correction_heads = []
        for _ in range(self.num_heads - 1):
            correction_head = deepcopy(template_head)
            # Temp fix, TBC more robust dimensions later on 
            correction_head.perm_decoder = DifferenceDecoder(ground_dim=8, 
                                                             excited_dim=8, 
                                                             hidden_dim=128, 
                                                             n_energies=3)
            _initialise_correction_head(correction_head)
            correction_heads.append(correction_head)
            

        transfer_model.autoencoder_heads = torch.nn.ModuleList(
            [template_head] + correction_heads
        )   

        # Freeze the autoencoder heads based on LR 
        for head, multiplier in zip(
            transfer_model.autoencoder_heads,
            self._head_lr_vector
        ):
            # Check if the multiplier has LR of 0, if so then make require gradients false
            if multiplier == 0.0:
                for parameter in head.parameters():
                    parameter.requires_grad = False

        # Re-register the buffer attribute for the updated learning rates 
        # This will be read by the optimiser constructor later on
        transfer_model.lr_multipliers = transfer_model.lr_multipliers.new_tensor(
            [self.gnn_lr, *self._head_lr_vector]
        )

        # Replace the e0s with the new metadata e0s
        # Preserve dtype and device
        current_e0s = transfer_model.atomic_energies_fn.atomic_energies
        transfer_model.atomic_energies_fn.atomic_energies = torch.as_tensor(
            self.metadata.atomic_energies,
            dtype=current_e0s.dtype,
            device=current_e0s.device,
        ).clone()

        # Define the routes
        route_matrix = torch.tril(
            torch.ones(
                self.num_heads, 
                self.num_heads,
                dtype=transfer_model.head_routes.dtype,
                device=transfer_model.head_routes.device
            )
        )
        transfer_model.head_routes = route_matrix

        return transfer_model


@dataclass
class MultiHeadStrategy:
    """
    Duplicate a trained autoencoder head for multi-head training
    This trainer will re train the HF data rather than treat
    it as a correction
    """

    metadata: AtomDataMetadata
    gnn_lr: float = 0.01
    head_multipliers: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.metadata.num_heads < 2:
            raise ValueError("metadata must contain at least 2 heads.")

        expected_shape = (self.metadata.num_heads, self.metadata.num_elements)
        if self.metadata.atomic_energies.shape != expected_shape:
            raise ValueError(
                f"metadata.atomic_energies must have shape {expected_shape}."
            )

        self.num_heads = self.metadata.num_heads

        if self.head_multipliers is None:
            self._head_lr_vector = [0.01] + [1.0] * (self.num_heads - 1)
        else:
            if set(self.head_multipliers) != set(self.metadata.head_to_index):
                raise ValueError(
                    "head_multipliers must have the same head keys as metadata head to index"
                )

            self._head_lr_vector = [
                float(self.head_multipliers[head_name])
                for head_name in self.metadata.head_to_index
            ]

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        transfer_model = _copy_model(model)

        if len(transfer_model.autoencoder_heads) != 1:
            raise ValueError(
                "MultiHeadStrategy expects a model with one template head."
            )

        # Add in the multiple Heads
        template_head = transfer_model.autoencoder_heads[0]
        transfer_model.autoencoder_heads = torch.nn.ModuleList(
            [template_head]
            + [deepcopy(template_head) for _ in range(self.num_heads - 1)]
        )

        for head, multiplier in zip(
            transfer_model.autoencoder_heads,
            self._head_lr_vector,
        ):
            if multiplier == 0.0:
                for parameter in head.parameters():
                    parameter.requires_grad_(False)

        transfer_model.lr_multipliers = transfer_model.lr_multipliers.new_tensor(
            [self.gnn_lr, *self._head_lr_vector]
        )

        # Replace the e0s with the new metadata e0s
        # Preserve dtype and device
        current_e0s = transfer_model.atomic_energies_fn.atomic_energies
        transfer_model.atomic_energies_fn.atomic_energies = torch.as_tensor(
            self.metadata.atomic_energies,
            dtype=current_e0s.dtype,
            device=current_e0s.device,
        ).clone()

        # Define the routes
        route_matrix = torch.eye(
            self.num_heads,
            dtype=transfer_model.head_routes.dtype,
            device=transfer_model.head_routes.device
        )
        transfer_model.head_routes = route_matrix

        return transfer_model
