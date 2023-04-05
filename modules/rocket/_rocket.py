from typing import Optional
from numpy.typing import NDArray

from dataclasses import dataclass
import multiprocessing

from .base import BaseTransformer

@dataclass
class Rocket(BaseTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET).

    Parameters
    ----------
    num_kernels: int = 100
        the number of kernels to apply to the data.
    kernel_dim: int = 1
        which dimension of kernels to use.
    random_state: Optional[int] = None
        random state for setting the seed.
    njobs: int = 1
        the number of cores to use for parallel computation.
    """
    num_kernels: int = 100
    kernel_dim: int = 1
    random_state: Optional[int] = None
    njobs: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.kernel_dim in [1, 2]:
            raise ValueError("Only 1 or 2 dimension kernels are available.")
        if self.njobs > multiprocessing.cpu_count():
            raise ValueError("njobs value exceeded the maximum cpu number.")
    
    def _fit(self, X: NDArray):
        """Generate random kernels adjusted to time series shape.

        Parameters
        ----------
        X : NDArray
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        
        Return
        ----------
        self
        """
        from ._kernels import _generate_1d_kernels, _generate_nd_kernels

        _, self.num_features, self.num_timepoints = X.shape

        if self.kernel_dim == 1:
            self.kernels = _generate_1d_kernels(self.num_features, self.num_timepoints, self.num_kernels, self.random_state)
        else:
            self.kernels = _generate_nd_kernels(self.num_features, self.num_timepoints, self.num_kernels, self.kernel_dim, self.random_state)

        return self
    
    def _transform(self, X: NDArray):
        """Transform input time series X with random kernels.

        Parameters
        ----------
        X : NDArray
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform.
        
        Return
        ----------
        out: NDArray[float32]
            transformed version of the input X with size of (num_instances, num_kernels*2).
        """
        from ._kernels import _apply_1d_kernels, _apply_nd_kernels
        from numba import get_num_threads, set_num_threads

        prev_threads = get_num_threads()
        set_num_threads(self.njobs)
        
        if self.kernel_dim == 1:
            out = _apply_1d_kernels(X, self.kernels)
        else:
            out = _apply_nd_kernels(X, self.kernels, self.kernel_dim)
        set_num_threads(prev_threads)
        return out


    
