from typing import Tuple
from numpy.typing import NDArray

import numpy as np
from scipy.linalg import inv
from scipy import stats
from dataclasses import dataclass
from joblib import Parallel, delayed

from util import *
from metrics import get_scorer

@dataclass
class RidgeRegression():
    """A linear guassian ridge model.

    x_t=fitted_param.T・n_t + b_t
    x_t: discretized position
    fitted_param: parameter
    n_t: spikes
    b_t: intercept
    """
    def __post_init__(self) -> None:
        self.fitted_param = None

    def fit(self, X_train: NDArray, y_train: NDArray, penalty: float):
        """Fitting based on training data.
        
        return the fitted coefficients

        Parameter:
        ---------
        X_train: NDArray
            train design matrix including one column full of 1 for the intercept
        y_train: NDArray
            discretized position from continuous coordinates to discrete value 1,2,3...
        penalty: float
            the penalty added on ridge model
        """
        self.X_train = X_train
        self.y_train = y_train.ravel()
        self.penalty = penalty

        self.xxlam = inv(self.X_train.T @ self.X_train + penalty * np.identity(self.X_train.shape[1])) # (X'X + lambda*I)^-1
        self.fitted_param = (self.xxlam @ self.X_train.T).dot(self.y_train)

    def predict(self, X_test: NDArray):
        """Predicting using fitted parameters based on test data.
        
        return the predicted results

        Parameter:
        ---------
        X_test: NDArray
            including one column full of 1 for the intercept

        """
        if self.fitted_param is None:
            raise ValueError("the model is not fitted, please call fit() first.")
        return X_test @ self.fitted_param

    def load(self, fitted_param: NDArray) -> None:
        self.fitted_param = fitted_param

    def evaluate(self, X_test: NDArray, y_test: NDArray, scoring: str) -> Tuple[NDArray]:
        if self.fitted_param is None:
            raise ValueError("fitted parameters are not loaded, please call load() first.")
        else:
            y_pred = X_test @ self.fitted_param
            scorer = get_scorer(scoring)
            test_scores = scorer(y_test, y_pred)
            return test_scores, y_pred

@dataclass
class SoftmaxRegression():
    """A softmax regression model.
    """
    def __post_init__(self) -> None:
        """Init.
        """

    def _one_hot(self, y: NDArray, n_classes: NDArray) -> NDArray:
        """Convert a vector into a one hot matrix.
        """
        self.y_hot = np.zeros((len(y), n_classes))
        self.y_hot[np.arange(len(y)), [int(x-1) for x in y]] = 1
        return self.y_hot

    def _likelihood_loss(self, y_hat) -> float:
        """A likelihood loss function for softmax regression.
        """
        loss = - np.sum(np.log(y_hat)[np.arange(len(self.y_train)), [int(x-1) for x in self.y_train]])
        return loss

    def _softmax_regression_path(self, iteration: int, lr: float, eps: float = .01):
        """Softmax regression fitting procedure.
        """
        z = self.X_train @ self.beta
        y_hat = softmax(z)
        
        grad = self.X_train.T.dot(y_hat - self.y_hot)
        self.beta = self.beta - lr * grad

        loss = self._likelihood_loss(y_hat)
        self.losses.append(loss)

        # print(f'Epoch {iteration+1}==> Loss = {loss}')

        if np.abs(self.losses[-1]-self.losses[-2]) < eps:
            return 
        return
    

    def fit(self, X_train: NDArray, y_train: NDArray , lr: float, max_iter: int = 1000, eps: float = .01) -> Tuple[NDArray]:
        """Fit the model.
        """
        self.X_train, self.y_train = X_train, y_train
        n, m = X_train.shape
        n_classes = len(np.unique(y_train))

        self.beta = np.random.random((m, n_classes)) # coeffs

        y_hot = self._one_hot(y_train, n_classes)

        self.losses = [np.inf]
        
        for iteration in range(max_iter):
            z = self.X_train @ self.beta
            y_hat = softmax(z)
            
            grad = self.X_train.T.dot(y_hat - self.y_hot)
            self.beta = self.beta - lr * grad

            loss = self._likelihood_loss(y_hat)
            self.losses.append(loss)

            # print(f'Epoch {iteration+1}==> Loss = {loss}')

            if np.abs(self.losses[-1] - self.losses[-2]) < eps:
                break
        
        return np.array(self.losses[1:]), self.beta

    def predict(self, X_test: NDArray, beta: NDArray):
        """Predict based on the fitted model.
        """
        z = X_test @ beta
        y_hat = softmax(z)
        
        return np.argmax(y_hat, axis=1)

        