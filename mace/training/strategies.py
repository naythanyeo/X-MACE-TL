"""
Model transformations for transfer-learning experiments.

Input will be a trained model, and output will be a new model
with the implemented strategies. Eg frozen layers etc
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch


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
class MultiHeadStrategy:
    """Duplicate a trained autoencoder head for multi-head training."""

    num_heads: int

    def __post_init__(self) -> None:
        if self.num_heads < 2:
            raise ValueError("num_heads must be at least 2.")

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        transfer_model = _copy_model(model)

        if len(transfer_model.autoencoder_heads) != 1:
            raise ValueError(
                "MultiHeadStrategy expects a model with one template head."
            )

        template_head = transfer_model.autoencoder_heads[0]
        transfer_model.autoencoder_heads = torch.nn.ModuleList(
            [template_head]
            + [deepcopy(template_head) for _ in range(self.num_heads - 1)]
        )

        return transfer_model
