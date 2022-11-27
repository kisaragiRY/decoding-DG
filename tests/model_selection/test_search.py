from modules.model_selection import SearchCV
from modules.decoder import RidgeRegression
from modules.metrics import get_scorer
from param import *

import pytest
import numpy as np

n_neurons=30
time_bins_train=100
time_bins_test=80
n_positions=4

@pytest.fixture
def train_set():
    np.random.seed(0)
    X = np.random.rand(time_bins_train,n_neurons)
    y = np.random.uniform(low=-40, high=40, size=(time_bins_train,1))
    return X, y

@pytest.mark.parametrize("train_index, test_index, hyper_param", 
                        [[range(35), range(35,36), 5],[range(99), range(99,100), 6]])
def test_fit_and_score(train_set, train_index, test_index, hyper_param):
    X ,y = train_set
    search = SearchCV(RidgeRegression(), ParamTrain().scoring, ParamTrain().penalty_range, 10)
    result = search.fit_and_score(X, y, train_index, test_index, hyper_param)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()

    rr = RidgeRegression()
    rr.fit(X_train, y_train, hyper_param)
    rr.predict(X_test)

    scorer = get_scorer(ParamTrain().scoring)
    train_scores = scorer(y_train, np.einsum("ij,j->i",X_train, rr.fitted_param))
    test_scores = scorer(y_test, rr.prediction)


    assert (train_scores == result["train_scores"]).all()
    assert (test_scores == result["test_scores"]).all()
    assert (rr.fitted_param == result["fitted_param"]).all()
