import pytest
import numpy as np
from Modules.decoder import linear_gaussian



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

def test_fit(train_set):
    lg=linear_gaussian()
    assert lg.fit(train_set[0],train_set[1]).any()

def test_predict(train_set,test_set):
    lg=linear_gaussian()
    lg.fit(train_set[0],train_set[1])
    assert lg.predict(test_set[0]).any()

