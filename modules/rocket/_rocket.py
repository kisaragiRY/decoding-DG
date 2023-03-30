from numpy.typing import NDArray

from dataclasses import dataclass

from base import BaseTransformer


class Rocket(BaseTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET).
    """
    num_kernels: int = 100
    random_state = None

    def __post_init__(self):
        super().__post_init__()
    
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
        from _kernels import _generate_kernels

        _, self.num_features, self.num_timepoints = X.shape
        self.kernels = _generate_kernels(self.num_features, self.num_timepoints, self.num_kernels, self.random_state)

        return self
    
    def _transform(self, X: NDArray):
        """Transform input time series X with random kernels.

        Parameters
        ----------
        X : NDArray
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        
        Return
        ----------
        transformed version of the input X.
        """
        from _kernels import _apply_kernels

        return _apply_kernels(X, self.kernels)


    
