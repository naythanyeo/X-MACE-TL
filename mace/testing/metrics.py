"""Helper functions to evaluate Tester errors."""
import numpy as np


def mean_absolute_error(predicted: np.ndarray, reference: np.ndarray, axis=None):
    return np.mean(np.abs(predicted - reference), axis=axis)


def phase_root_mean_squared_error(
    predicted: np.ndarray, reference: np.ndarray, axis=None
):
    negative_error2 = np.sum(np.square(reference - predicted), axis=-1)
    positive_error2 = np.sum(np.square(reference + predicted), axis=-1)
    phase_error2 = np.minimum(negative_error2, positive_error2)
    return np.sqrt(np.mean(phase_error2, axis=axis))
