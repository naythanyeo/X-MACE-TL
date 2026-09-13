"""
Helper functions to properly get the NAC phase loss
Loss for NACs must be grouped for each molecule and across electronic states

For each NAC pair, they can differ up to a sign based on the wavefunction convention
For 3 states, there are four unique electronic-state phase assignments.
The resulting NAC-vector signs must be consistent across state pairs.

First helper enumerates all valid electronic-state phase assignments.
For the data of one complete geometry, the next helper aligns them and
gets the sign pair combination with the minimum MSE 
Last helper deals with one batch, and calls the second helper for each geometry
Ensures that each molecule and electronic states are consistent
"""

from itertools import product
import numpy as np
import torch


def smooth_to_raw_nacs(
    smooth_nacs: torch.Tensor,
    energies: torch.Tensor,
    batch: torch.Tensor,
    gap_floor: float = 1e-8, # Follows current X MACE calculator default
) -> torch.Tensor:
    """
    Convert graph-level smooth NACs into atom-level raw NACs.
    """
    num_states = energies.shape[-1]
    state_i, state_j = torch.triu_indices(
        num_states,
        num_states,
        offset=1,
        device=energies.device
    )
    # Follow clamping for minimum value 
    energy_gaps = torch.abs(
        energies[:, state_j] - energies[:, state_i]
    ).clamp_min(gap_floor)
    # Expand based on number of atoms 
    node_energy_gaps = energy_gaps[batch]
    return smooth_nacs / node_energy_gaps.unsqueeze(-1)


def enumerate_nac_phase_signs(num_states):
    """
    We first find all unique electronic-state phase assignments.

    S0 is fixed to +1 because when you calculate the pair signs later it will be
    duplicated essentially 
    For 3 states s0 s1 s2, this will return 
    1 1 1
    1 1 -1
    1 -1 1
    1 -1 -1
    
    For each configuration, we then need to convert into signs for each pair
    Here we assume and expect that NAC pairs follow the same convention 
    (S0, S1), (S0, S2), (S1, S2) with np.triu_indices

    For each pair, we calculate the sign that we need to multiply it by
    """
    # Validation first
    if not isinstance(num_states, int) or isinstance(num_states, bool) or num_states < 1:
        raise ValueError("num_states must be a positive integer.")

    # Find the wfn configurations
    wfn_sign_configurations = np.array(
        [[1, *remaining_signs]
        for remaining_signs in product((1, -1), repeat=num_states - 1)]
    )

    # For each pair, get the multiplier sign
    pair_signs = []
    for pair in zip(*np.triu_indices(num_states, k=1)):
        # pair is (s0, s1) for eg
        # wfh_sign_configuraiton has rows of configurations [1, -1, 1] for eg
        # which each row represents s0 s1 s2
        signs = (
            wfn_sign_configurations[:, pair[0]] * 
            wfn_sign_configurations[:, pair[1]]
        )
        pair_signs.append(signs)

    # pair_signs is now [[s0s1 4configs], [s0s2 4configs], [s1s2 4configs]]
    # transpose so that it becomes [[config1_s0s1_s0s2_s1s2], [config2...]..]
    return np.array(pair_signs).T


def align_geometry_nacs(geom_nac_pred, geom_nac_ref, pair_signs):
    # This takes in nac_pred and nac_ref for ONE geometry
    # pair signs is by default calcualted at batch level, it will be same
    # device and dtype as the predictions
    # Returns the residual aftering aligning based on MSE
    best_MSE = None
    best_residue = None

    # Get the MSE for each config signs 
    for config_signs in pair_signs:
        # Rehshape config_signs to match dimension of the predictions 
        # predictions of dimension [n_atoms, nac_pairs, xyz]
        # signs should be multiplied to the middle dimension
        pred_transformed = geom_nac_pred*config_signs[None, :, None]
        config_MSE = torch.mean(
            torch.square(
                pred_transformed - geom_nac_ref
            )
        )
        
        if best_MSE is None or config_MSE < best_MSE:
            best_MSE = config_MSE
            config_residual = pred_transformed - geom_nac_ref
            best_residue = config_residual

    return best_residue
        

def align_batch_nacs(pred, ref, ptr, num_states):
    # Pred and Ref are one whole batch NAC data
    # They are assigned in trainer or loss as pred["smooth_nacs"]
    # Ptr is geometry boundaries 
    # Returns best aligned residue of this batch pred and ref
    
    # First return pair_signs
    pair_signs = pred.new_tensor(
        enumerate_nac_phase_signs(num_states)
    )

    geometry_residuals = []

    # Enumerate through all the geometries 
    for geometry_index in range(ptr.numel() - 1):
        start = int(ptr[geometry_index].item())
        end = int(ptr[geometry_index + 1].item())

        geometry_residuals.append(
            align_geometry_nacs(
                pred[start:end],
                ref[start:end],
                pair_signs
            )
        )

    # Return the residual of the geometries in that batch
    return torch.cat(geometry_residuals, dim=0)
