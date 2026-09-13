"""Helper functions to evaluate Tester errors."""
import numpy as np
import torch

from mace.modules.nac_utils import align_batch_nacs


def mean_absolute_error(predicted: np.ndarray, reference: np.ndarray, axis=None):
    return np.mean(np.abs(predicted - reference), axis=axis)


def phase_mean_absolute_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    ptr: np.ndarray,
    num_states: int,
    axis=None,
):
    residue = align_batch_nacs(
        pred=torch.as_tensor(predicted),
        ref=torch.as_tensor(reference),
        ptr=torch.as_tensor(ptr),
        num_states=num_states,
    )
    return np.mean(np.abs(residue.numpy()), axis=axis)
