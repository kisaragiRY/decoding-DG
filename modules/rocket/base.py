from numpy.typing import NDArray

from dataclasses import dataclass

@dataclass
class BaseTransformer():
    """Transformer base class.
    """
    def __post_init__(self):
        """Post init.
        """
        self._is_fitted = False

    def check_is_fitted(self):
        """Check if the estimator has been fitted.
        Raises
        ------
        ValueError
            If the estimator has not been fitted yet.
        """
        if not self._is_fitted:
            raise ValueError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

    def fit(self, X: NDArray):
        """Fit transformer to X.

        Parameters
        ----------
        X : NDArray
            time series after segmentation.
            The format of the data should be a 3D NDArray which
            is of shape (n_instances, n_features, n_timepoints)

        Return
        ----------
        self
            a fitted instance.
        """
        self._fit(X=X)
        self._is_fitted = True
        return self
    
    def transform(self, X: NDArray, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : NDArray
            time series after segmentation.
            The format of the data should be a 3D NDArray which
            is of shape (n_instances, n_features, n_timepoints)

        Return
        ----------
        transformed version of X
        """
        self.check_is_fitted()

        return self._transform(X)
    
    def fit_transform(self, X: NDArray, y=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : NDArray
            time series after segmentation.
            The format of the data should be a 3D NDArray which
            is of shape (n_instances, n_features, n_timepoints)

        Return
        ----------
        transformed version of X
        """
        return self.fit(X).transform(X)


    def _fit(self, X: NDArray, y=None):
        """Fit transformer to X.

        Parameters
        ----------
        X : NDArray
            time series after segmentation.
            The format of the data should be a 3D NDArray which
            is of shape (n_instances, n_features, n_timepoints)

        Return
        ----------
        self
            a fitted instance.
        """

    def _transform(self, X: NDArray, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : NDArray
            time series after segmentation.
            The format of the data should be a 3D NDArray which
            is of shape (n_instances, n_features, n_timepoints)

        Return
        ----------
        transformed version of X
        """





    