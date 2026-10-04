"""
Model transformations for transfer-learning experiments.

Input will be a trained model, and output will be a new model
with the implemented strategies. Eg frozen layers etc
"""

from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatchcase
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


freeze_layers: Dict[str, Tuple[str, ...]] = {
    "full_graph": ("node_embedding", "radial_embedding", "interactions", "products"),
    "node": ("node_embedding",),
    "interactions": ("interactions",),
    "products": ("products",),
    "encoder": ("autoencoder_heads.*.perm_encoder",),
    "decoder": ("autoencoder_heads.*.perm_decoder",),
    "invariant_readouts": ("autoencoder_heads.*.invariant_readouts",),
    "nac_readouts": ("autoencoder_heads.*.nac_readouts",),
    "soc_readouts": ("autoencoder_heads.*.socs_readouts",),
    "readouts": (
        "autoencoder_heads.*.invariant_readouts",
        "autoencoder_heads.*.nac_readouts",
        "autoencoder_heads.*.socs_readouts",
    ),
    "all_heads": ("autoencoder_heads",),
    "head0": ("autoencoder_heads.0",),
}

lora_layer_aliases: Dict[str, Tuple[str, ...]] = {
    # RadialEmbeddingBlock currently has no injectable trainable linear layers.
    "full_graph": ("node_embedding", "interactions", "products"),
    "node": ("node_embedding",),
    "interactions": ("interactions",),
    "products": ("products",),
    "encoder": ("autoencoder_heads.*.perm_encoder",),
    "decoder": ("autoencoder_heads.*.perm_decoder",),
    "invariant_readouts": ("autoencoder_heads.*.invariant_readouts",),
    "nac_readouts": ("autoencoder_heads.*.nac_readouts",),
    "soc_readouts": ("autoencoder_heads.*.socs_readouts",),
    "readouts": (
        "autoencoder_heads.*.invariant_readouts",
        "autoencoder_heads.*.nac_readouts",
        "autoencoder_heads.*.socs_readouts",
    ),
    "all_heads": ("autoencoder_heads",),
    "head0": ("autoencoder_heads.0",),
}


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
        if not all(isinstance(layer, str) and layer for layer in self.frozen_layers):
            raise TypeError("Each frozen layer must be a non-empty string.")
        if len(self.frozen_layers) != len(set(self.frozen_layers)):
            raise ValueError("frozen_layers cannot contain duplicate entries.")

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        transfer_model = _copy_model(model)

        modules = dict(transfer_model.named_modules())
        parameters = dict(transfer_model.named_parameters())
        parameters_to_freeze = {}

        for layer in self.frozen_layers:
            # Check the pre-defined freeze_layers
            # Prefered input because these are readable names
            if layer in freeze_layers:
                module_patterns = freeze_layers[layer]
                module_names = {
                    name
                    for pattern in module_patterns
                    for name in modules
                    if fnmatchcase(name, pattern)
                }
                matched_parameters = {
                    name: parameter
                    for name, parameter in parameters.items()
                    if parameter.requires_grad
                    and any(
                        name == module_name or name.startswith(f"{module_name}.")
                        for module_name in module_names
                    )
                }
            # Check the modules for exact-name matches
            elif layer in modules:
                matched_parameters = {
                    name: parameter
                    for name, parameter in parameters.items()
                    if parameter.requires_grad
                    and (name == layer or name.startswith(f"{layer}."))
                }
            # Also check the paramters for exact-name matches
            elif layer in parameters:
                parameter = parameters[layer]
                matched_parameters = {layer: parameter} if parameter.requires_grad else {}
            else:
                aliases = ", ".join(sorted(freeze_layers))
                raise ValueError(
                    f"Unknown frozen layer '{layer}'. Accepted aliases: {aliases}. "
                    "You can also use an exact module path from model.named_modules() "
                    "or parameter name from model.named_parameters()."
                )

            if not matched_parameters:
                raise ValueError(
                    f"Frozen layer '{layer}' did not match any trainable parameters."
                )
            parameters_to_freeze.update(matched_parameters)

        for parameter in parameters_to_freeze.values():
            parameter.requires_grad_(False)

        if not any(parameter.requires_grad for parameter in transfer_model.parameters()):
            raise ValueError("FreezeStrategy cannot freeze every remaining trainable parameter.")

        return transfer_model


@dataclass
class LoRAStrategy:
    """Inject trainable adapters, freezing only their original linear layers."""

    rank: int = 4
    alpha: float = 1.0
    lora_layers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Validate rank
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer.")
        if self.rank <= 0:
            raise ValueError("rank must be greater than zero.")

        # Validate alpha
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a real number.")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and greater than zero.")

        # Validate lora_layers format
        if not isinstance(self.lora_layers, (tuple, list)):
            raise TypeError("lora_layers must be a tuple or list of layer names.")
        if not self.lora_layers:
            raise ValueError("lora_layers must contain at least one layer.")
        for layer in self.lora_layers:
            if not isinstance(layer, str):
                raise TypeError("Each LoRA layer name must be a string.")
            if not layer:
                raise ValueError("LoRA layer names must not be empty.")

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        # First deep copy the original model
        transfer_model = deepcopy(model)
        modules = dict(transfer_model.named_modules())

        for layer in self.lora_layers:
            # Check the pre-defined user facing lora layer aliases
            # Preferred input as its simpler terms
            if layer in lora_layer_aliases:
                module_patterns = lora_layer_aliases[layer]
                paths = [
                    name for name in modules
                    if any(fnmatchcase(name, pattern) for pattern in module_patterns)
                ]

            # Check for exact-name matches in the modules also
            # Parameters not considered because LORA requires a module to inject into
            elif layer in modules:
                paths = [layer]
            else:
                aliases = ", ".join(sorted(lora_layer_aliases))
                raise ValueError(
                    f"Unknown LoRA layer '{layer}'. Accepted aliases: {aliases}. "
                    "You can also use an exact module path from model.named_modules()."
                )

            if not paths:
                raise ValueError(f"LoRA layer '{layer}' did not match any modules.")

            for path in paths:
                # For each of the paths, get the modules
                module = transfer_model.get_submodule(path)
                # Insert lora wrapped layer with the previous helper 
                replacement = inject_lora(module, rank=self.rank, alpha=self.alpha)

                # Check which of the children are lora wrapped 
                wrappers = [
                    child for child in replacement.modules()
                    if isinstance(child, (LoRAO3Linear, LoRADenseLinear, LoRAFCLayer))
                ]

                # If not comptaible lora layers, then raise error
                if not wrappers:
                    raise ValueError(
                        f"LoRA layer '{path}' contains no compatible linear layers."
                    )

                # Freeze the base layers of the lora wrappers 
                for wrapper in wrappers:
                    wrapper.base.requires_grad_(False)

                parent_path, _, child_name = path.rpartition(".")
                setattr(transfer_model.get_submodule(parent_path), child_name, replacement)

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
