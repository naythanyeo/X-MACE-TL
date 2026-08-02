"""Helper functions to evaluate Tester errors."""
import numpy as np


def mean_absolute_error(predicted: np.ndarray, reference: np.ndarray, axis=None):
    return np.mean(np.abs(predicted - reference), axis=axis)
