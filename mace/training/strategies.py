"""
Model transformations for transfer-learning experiments.

Input will be a trained model, and output will be a new model
with the implemented strategies. Eg frozen layers etc
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Tuple

import torch

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

        return transfer_model
