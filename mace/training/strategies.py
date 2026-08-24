"""
Model transformations for transfer-learning experiments.

Input will be a trained model, and output will be a new model
with the implemented strategies. Eg frozen layers etc
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from e3nn import o3

from mace.data.atom_data_loader import AtomDataMetadata
from mace.modules.blocks import (
    DifferenceDecoder
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
