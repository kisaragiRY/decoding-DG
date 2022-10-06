import pytest
import numpy as np
from modules.infoMetrics import InfoMetrics

n_neurons=30
time_bins_train=100
time_bins_test=80
n_positions=4

@pytest.fixture
def data_set():
    np.random.seed(0)
    spikes= np.random.rand(time_bins_train,n_neurons)
    binned_position_train=np.random.randint(0,n_positions+1,size=(time_bins_train,1))
    return spikes,binned_position_train

def test_cal_mi(data_set):
    """Test cal_mutual_info() in InfoMetrics class.
    """
    info_metrics=InfoMetrics(spikes=data_set[0],status=data_set[1])
    assert info_metrics.cal_mutual_info(0)