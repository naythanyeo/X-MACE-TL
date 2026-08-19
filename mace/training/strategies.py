"""
Model transformations for transfer-learning experiments.

Input will be a trained model, and output will be a new model
with the implemented strategies. Eg frozen layers etc
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
from e3nn import o3

from mace.data.atom_data_loader import AtomDataMetadata


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


def _initialise_decoder_output(
    decoder: torch.nn.Module, scale: float = 1e-4
) -> None:
    output_layer = decoder.decoder_nn[-1]

    with torch.no_grad():
        output_layer.weight.normal_(mean=0.0, std=scale)
        output_layer.bias.zero_()

        diagonal_bias = torch.linspace(
            -scale,
            scale,
            decoder.n_energies,
            dtype=output_layer.bias.dtype,
            device=output_layer.bias.device,
        )
        output_layer.bias[:decoder.n_energies].copy_(diagonal_bias)


def _initialise_correction_head(head: torch.nn.Module) -> None:
    """
    Helper to initialise the correction head instead of just deep copying it
    Makes it such that the initial outputs are almost 0 so the model starts closer 
    to the actual values. Else the initial input will be almost double the errors
    """
    _reset_module_parameters(head)
    _initialise_decoder_output(head.perm_decoder)

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
    GNN Lr is taken to be the base LR for all
    Base and correction head LR are relative to the GNN LR
    """

    metadata: AtomDataMetadata
    base_head_lr: float = 0.1
    correction_head_lr: float = 10

    def __post_init__(self) -> None:
        if self.metadata.num_heads < 2:
            raise ValueError("metadata must contain at least 2 heads.")

        expected_shape = (self.metadata.num_heads, self.metadata.num_elements)
        if self.metadata.atomic_energies.shape != expected_shape:
            raise ValueError(
                f"metadata.atomic_energies must have shape {expected_shape}."
            )

        self.num_heads = self.metadata.num_heads

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        transfer_model = _copy_model(model)

        if len(transfer_model.autoencoder_heads) != 1:
            raise ValueError(
                "MultiHeadStrategy expects a model with one template head."
            )

        template_head = transfer_model.autoencoder_heads[0]
        correction_heads = []
        for _ in range(self.num_heads - 1):
            correction_head = deepcopy(template_head)
            _initialise_correction_head(correction_head)
            correction_heads.append(correction_head)

        transfer_model.autoencoder_heads = torch.nn.ModuleList(
            [template_head] + correction_heads
        )   

        # Fill in the learning rates
        for module in transfer_model.autoencoder_heads[0].modules():
            module.lr_multiplier.fill_(self.base_head_lr)
        for i in range(1, self.num_heads):
            for module in transfer_model.autoencoder_heads[i].modules():
                module.lr_multiplier.fill_(self.correction_head_lr)

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

    def __post_init__(self) -> None:
        if self.metadata.num_heads < 2:
            raise ValueError("metadata must contain at least 2 heads.")

        expected_shape = (self.metadata.num_heads, self.metadata.num_elements)
        if self.metadata.atomic_energies.shape != expected_shape:
            raise ValueError(
                f"metadata.atomic_energies must have shape {expected_shape}."
            )

        self.num_heads = self.metadata.num_heads

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
