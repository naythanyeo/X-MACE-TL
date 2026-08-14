"""
Model transformations for transfer-learning experiments.

Input will be a trained model, and output will be a new model
with the implemented strategies. Eg frozen layers etc
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch

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
