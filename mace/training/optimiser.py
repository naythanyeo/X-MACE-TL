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

    # First build all the multiplier terms for the learning rate 
    # Currently info is being stored in a vector model.lr_multipliers such that
    # [gnn_lr, head0_lr, head1_lr ...]
    # Later on for every module and parameter, a parameter_groups dict is built to specified
    # details like the decay and LR. So we first build a dictionary that relates this index
    # vector to the exact module names that can be accesssed later

    module_multipliers = {}

    for module_name, _ in model.named_modules():
        # If the module is part of the head, then module name will start with autoencoder_heads
        # This will be starting from index 1 onwards 
        if module_name.startswith("autoencoder_heads."):
            # Get the index of which head it belongs to 
            # The autoencoder head index will be after the first "."
            head_index = int(module_name.split(".")[1])
            multiplier = float(
                model.lr_multipliers[head_index+1].item() # First index is the GNN so +1 
            )
        else:
            # If not this is the graph layer, we follow the LR of the first term
            multiplier = float(
                model.lr_multipliers[0].item()
            )
        # Now add in the multipliers to the module_multiplier dictionary with module names as keys
        module_multipliers[module_name] = multiplier

    parameter_groups = {}

    for module_name, module in model.named_modules():
        multiplier = module_multipliers[module_name]
        effective_lr = lr * multiplier
        
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or effective_lr == 0:
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
