"""Metrics to assess performance on regression tasks."""

import numpy as np

__ALL__ = [
    "mean_square_error",
]

def _check_consistent_length(*arrays):
    """Check all the arrays have the same length."""
    lengths = [len(x) for x in arrays]
    if len(np.unique(lengths)) > 1:
        raise ValueError("Found input variables have inconsistent length.")

def mean_square_error(y_true: np.array,y_pred: np.array) -> float:
    """Calculate the mean square error for y_true and y_pred."""
    _check_consistent_length(y_true, y_pred)
    errors = np.average((y_true.ravel() - y_pred.ravel())**2)
    return errors
