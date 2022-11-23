import pytest
import numpy as np
from modules.func import *

n_neurons=30
time_bins_train=100
time_bins_test=80
n_positions=4

@pytest.fixture
def train_set():
    np.random.seed(0)
    spikes = np.random.rand(time_bins_train,n_neurons)
    coordinate = np.random.randint(0,n_positions+1,size=(time_bins_train,1))
    return spikes,coordinate 

def test_design_matrix_decoder1(train_set):
    """Test function mk_design_matrix_decoder.

    Test wit different nthist values
    """
    nthist=2
    design_matrix=mk_design_matrix_decoder1(train_set[0],nthist)
    assert (design_matrix.shape[1]-1)/nthist==n_neurons

    nthist=0
    design_matrix=mk_design_matrix_decoder1(train_set[0],nthist)
    assert (design_matrix.shape[1]-1)==n_neurons

@pytest.mark.parametrize("nthist",[1])
def test_design_matrix_decoder3(train_set, nthist):
    """Test function mk_design_matrix_decoder3.

    See whether the output is correct.
    """
    spikes, coordinate = train_set
    design_matrix=mk_design_matrix_decoder3(spikes, coordinate, nthist)
    assert (spikes[nthist:] == design_matrix[:,1:-1]).all() # for spikes
    assert (coordinate[:-nthist].ravel() == design_matrix[:,-1].ravel()).all() # for coordinate