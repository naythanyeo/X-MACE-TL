import torch
from copy import deepcopy
from mace.tools.utils import AtomicNumberTable


def load_foundations(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=False,
    use_scale=True,
    max_L=2,
):
    """
    Load the foundations of a model into a model for fine-tuning.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)
    max_ell = model.spherical_harmonics._lmax  # pylint: disable=protected-access
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )
    if model.radial_embedding.bessel_fn.__class__.__name__ == "BesselBasis":
        model.radial_embedding.bessel_fn.bessel_weights = torch.nn.Parameter(
            model_foundations.radial_embedding.bessel_fn.bessel_weights.clone()
        )

    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors
        for j in range(4):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        )
                        .weight[:num_radial, :]
                        .clone()
                    )
                )
            else:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        ).weight.clone()
                    )
                )

        model.interactions[i].linear.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear.weight.clone()
        )
        if (
            model.interactions[i].__class__.__name__
            == "RealAgnosticResidualInteractionBlock"
        ):
            skip_tp_weight = (
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    num_species_foundations,
                    num_channels_foundation,
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
            if model.interactions[i].skip_tp.weight.shape == skip_tp_weight.shape:
                model.interactions[i].skip_tp.weight = torch.nn.Parameter(skip_tp_weight)
        else:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    (max_ell + 1),
                    num_species_foundations,
                    num_channels_foundation,
                )[:, :, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
    # Transferring products
    for i in range(2):  # Assuming 2 products modules
        max_range = max_L + 1 if i == 0 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            model.products[i].symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights, :, :]
                    .clone()
                )
            )

            for k in range(2):  # Assuming 2 weights in each contraction
                model.products[i].symmetric_contractions.contractions[j].weights[k] = (
                    torch.nn.Parameter(
                        model_foundations.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights[k][indices_weights, :, :]
                        .clone()
                    )
                )

        product_linear_weight = (
            model_foundations.products[i].linear.weight.detach().clone()
        )
        target_linear_weight = model.products[i].linear.weight

        # If the product paramters match, can directly copy over 
        if target_linear_weight.shape == product_linear_weight.shape:
            model.products[i].linear.weight = torch.nn.Parameter(
                product_linear_weight
            )
        # Special case where we just added a new vector channel for NACs 
        # in the second interaction block. ANI only has 192x0e for the 
        # second layer (original MACE only preserved scalar channel)
        # For NAC training we preserved 192x0e+192x1o+192x2e 
        elif (
            # Check that the second layer has 3x number of paramters 
            # Technically could work for other use case but we restrict
            # to this specific model params JIC 
            # For other model types it will just skip if its not a match
            i == 1
            and target_linear_weight.numel()
            == 3 * product_linear_weight.numel()
        ):
            # Target weights have the shape of the 1e 2o channels
            target_vector_tensor_weights = (
                target_linear_weight[product_linear_weight.numel():]
                .detach()
                .clone()
            )
            # Concat both layers then transfer to model
            model.products[i].linear.weight = torch.nn.Parameter(
                torch.cat(
                    (
                        product_linear_weight,
                        target_vector_tensor_weights,
                    )
                )
            )
    
    model.scale_shift = model_foundations.scale_shift

    return model
