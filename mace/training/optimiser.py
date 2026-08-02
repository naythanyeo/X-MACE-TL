"""
Construct optimisers with X-MACE parameter-group defaults
In this case use Adam with selective weight decay for interaction and product
blocks only. From the CLI, only these two blocks have weight decay applied
Everything else has 0 weight decay. Specifically for the interaction blocks,
only linear.weight and skip_tp_full.weight have weight decay applied

Other ADAM hyperparamters are just set to the defaults, ie 
betas=(0.9, 0.999), amsgrad=True
"""

import torch


def build_optimiser(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 5e-7,
) -> torch.optim.Adam:

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Follow CLI convention
        interaction_decay = name.startswith("interactions.") and (
            "linear.weight" in name or "skip_tp_full.weight" in name
        )
        if name.startswith("products.") or interaction_decay:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [
        {
            "name": "decay",
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "name": "no_decay",
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.Adam(
        parameter_groups,
        lr=lr,
        betas=(0.9, 0.999),
        amsgrad=True,
    )
