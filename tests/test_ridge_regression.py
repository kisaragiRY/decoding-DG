from unittest import result
import pytest
import numpy as np
from modules.decoder import RidgeRegression,Results

n_neurons=30
time_bins_train=100
time_bins_test=80
n_positions=4

@pytest.fixture
def train_set():
    np.random.seed(0)
    design_matrix_train= np.random.rand(time_bins_train,n_neurons)
    binned_position_train=np.random.randint(0,n_positions+1,size=(time_bins_train,1))
    return design_matrix_train,binned_position_train

@pytest.fixture
def test_set():
    np.random.seed(0)
    design_matrix_test= np.random.rand(time_bins_test,n_neurons)
    binned_position_test=np.random.randint(0,n_positions+1,size=(time_bins_test,1))
    return design_matrix_test,binned_position_test

def test_fit(train_set,test_set):
    rr=RidgeRegression()
    rr.fit(train_set[0],train_set[1],6)
    rr.predict(test_set[0])
    results=Results(rr)
    assert results.summary()

# def test_predict(train_set,test_set):
#     rr=RidgeRegression()
#     rr.fit(train_set[0],train_set[1],penalty=1)
#     assert rr.predict(test_set[0]).any()
