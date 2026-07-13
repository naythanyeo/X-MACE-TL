"""
Helper functions to evaluate error for tester. 
Only two basic ones so far 
"""
import numpy as np

def mean_absolute_error(predicted: np.ndarray, reference: np.ndarray):
    return np.mean(np.abs(predicted - reference))

def root_mean_squared_error(predicted: np.ndarray, reference: np.ndarray):
    return np.sqrt(np.mean((predicted - reference) ** 2))
