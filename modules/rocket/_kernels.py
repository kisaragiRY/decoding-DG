from typing import Tuple
from numpy.typing import NDArray

from dataclasses import dataclass

import numpy as np

def _generate_kernels(num_featurns: int, num_timepoints: int, num_kernels: int, seed: None):
    """Generate random kernels.
    """
    if seed is not None:
        np.random.seed(seed)
    
    

def _apply_kernels(X: NDArray, kernels: Tuple):
    """Apply the kernels to X.
    """

