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

def _uses_weight_decay(name: str) -> bool:
    interaction_decay = name.startswith("interactions.") and (
        "linear.weight" in name or "skip_tp_full.weight" in name
    )
    return name.startswith("products.") or interaction_decay


def build_optimiser(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 5e-7,
) -> torch.optim.Adam:

    parameter_groups = {}

    # Iterate through all the modules including child modules
    for module_name, module in model.named_modules():
        effective_lr = lr * module.lr_multiplier.item()
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            # Gives the full name including the module class at the start even when iterating through module
            full_name = (
                f"{module_name}.{parameter_name}"
                if module_name
                else parameter_name
            )
            decay = weight_decay if _uses_weight_decay(full_name) else 0

            parameter_group_key = f"{str(effective_lr)}_{str(decay)}"

            if parameter_group_key not in parameter_groups:
                parameter_groups[parameter_group_key] = {
                    "params": [],
                    "lr": effective_lr,
                    "weight_decay": decay
                }
            # Add actual paramters into the dict
            parameter_groups[parameter_group_key]["params"].append(parameter)

    return torch.optim.Adam(
        list(parameter_groups.values()),
        betas=(0.9, 0.999),
        amsgrad=True,
    )
