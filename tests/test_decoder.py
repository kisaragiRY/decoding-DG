from modules.decoder import RidgeRegression

import pytest
import numpy as np
from scipy.linalg import inv

@pytest.fixture
def train_set():
    n_neurons=30
    time_bins=100
    np.random.seed(0)
    X = np.random.rand(time_bins,n_neurons)
    y = np.random.uniform(low=-40, high=40, size=(time_bins,1))
    return X, y

@pytest.mark.parametrize("penalty", 
                        [[5]])
def test_RidgeRegression_fit(train_set, penalty):
    X, y = train_set
    xxlam = inv(X.T @ X + penalty * np.identity(X.shape[1])) # (X'X + lambda*I)^-1
    fitted_param = (xxlam @ X.T).dot(y.ravel())

    rr = RidgeRegression()
    rr.fit(X , y , penalty)

    assert (fitted_param == rr.fitted_param).all()