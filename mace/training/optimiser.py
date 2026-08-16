"""
Construct optimisers with X-MACE parameter-group defaults
In this case use Adam with selective weight decay for interaction and product
blocks only. From the CLI, only these two blocks have weight decay applied
Everything else has 0 weight decay. Specifically for the interaction blocks,
only linear.weight and skip_tp_full.weight have weight decay applied

Other ADAM hyperparamters are just set to the defaults, ie 
betas=(0.9, 0.999), amsgrad=True
"""

from typing import Optional

import torch


def _learning_rate_group(name: str) -> str:
    if name.startswith("autoencoder_heads.0."):
        return "base_head"
    if name.startswith("autoencoder_heads."):
        return "new_head"
    return "gnn"


def _uses_weight_decay(name: str) -> bool:
    interaction_decay = name.startswith("interactions.") and (
        "linear.weight" in name or "skip_tp_full.weight" in name
    )
    return name.startswith("products.") or interaction_decay


def build_optimiser(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 5e-7,
    base_head_lr: Optional[float] = None,
    new_head_lr: Optional[float] = None,
) -> torch.optim.Adam:
    learning_rates = {
        "gnn": lr,
        "base_head": lr if base_head_lr is None else base_head_lr,
        "new_head": lr if new_head_lr is None else new_head_lr,
    }
    parameter_groups = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate_group = _learning_rate_group(name)
        decay_group = "decay" if _uses_weight_decay(name) else "no_decay"
        group_name = f"{learning_rate_group}_{decay_group}"

        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                "name": group_name,
                "params": [],
                "lr": learning_rates[learning_rate_group],
                "weight_decay": weight_decay if decay_group == "decay" else 0.0,
            }
        parameter_groups[group_name]["params"].append(parameter)

    return torch.optim.Adam(
        list(parameter_groups.values()),
        betas=(0.9, 0.999),
        amsgrad=True,
    )
