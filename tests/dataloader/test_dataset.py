from numpy.typing import NDArray

import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from util import gauss1d
from modules.dataloader import *

@pytest.fixture
def data_dir():
    DATA_ROOT = Path('data/alldata/')
    data_dir = next(iter(DATA_ROOT.iterdir()))
    return data_dir

@pytest.mark.parametrize("coord_axis, nthist", [["x-axis", 1], ["y-axis", 3]])
def test_PastCoordDataset(data_dir, coord_axis, nthist):
    design_matrix, coord = PastCoordDataset(data_dir, False).load_all_data(coord_axis, nthist)

    coords_df = pd.read_csv(data_dir/'position.csv', index_col=0)
    ori_coord = coords_df.values[3:,1:3][:,0] if coord_axis == "x-axis" else coords_df.values[3:,1:3][:,1]# only take the X,Y axis data

    assert (ori_coord[nthist:].ravel() == coord.ravel()).all()
    assert (design_matrix[:,-1] == ori_coord[:-nthist]).all()

@pytest.mark.parametrize("coord_axis, nthist", [["x-axis", 0], ["x-axis", 1], ["y-axis", 2]])
def test_SpikesCoordDataset(data_dir, coord_axis, nthist):
    design_matrix, coord = SpikesCoordDataset(data_dir, False).load_all_data(coord_axis, nthist)

    coords_df = pd.read_csv(data_dir/'position.csv', index_col=0)
    ori_coord = coords_df.values[3:,1:3][:,0] if coord_axis == "x-axis" else coords_df.values[3:,1:3][:,1]# only take the X,Y axis data

    spikes_df = pd.read_csv(data_dir/'traces.csv', index_col=0)
    ori_spikes = spikes_df.values # only take the X,Y axis data

    if nthist != 0:
        assert (design_matrix[:,-1] == ori_coord[:-nthist]).all()
        assert (design_matrix[:,1:-1] == ori_spikes[nthist:]).all()
    else: 
        assert (design_matrix[:,1:] == ori_spikes[nthist:]).all()
    assert (ori_coord[nthist:] == coord).all()

@pytest.mark.parametrize(
    "coord_axis, nthist, window_size", 
    [
        ["x-axis", 0, 3], 
        ["x-axis", 1, 3], 
        ["y-axis", 0, 30],
        ["y-axis", 1, 30],
        ["y-axis", 0, 3],
        ["y-axis", 3, 3]
    ])
def test_SmoothedSpikesDataset(data_dir, coord_axis, nthist, window_size):
    """Test SmoothedSpikesDataset."""
    def filter_spikes(window_size: int, design_spikes: NDArray) -> NDArray:
        """Filter spikes with the given kernel."""
        kernel = gauss1d(np.linspace(-3, 3, window_size))

        def filtered(x: NDArray) -> NDArray:
            """Convovle with the given kernel."""
            return np.convolve(x, kernel, mode="same")

        return np.apply_along_axis(filtered, 0, design_spikes)

    design_matrix, coord = SmoothedSpikesDataset(data_dir, False).load_all_data(coord_axis, nthist, window_size)

    coords_df = pd.read_csv(data_dir/'position.csv', index_col=0)
    ori_coord = coords_df.values[3:,1:3][:,0] if coord_axis == "x-axis" else coords_df.values[3:,1:3][:,1]# only take the X,Y axis data

    spikes_df = pd.read_csv(data_dir/'traces.csv', index_col=0)
    ori_spikes = spikes_df.values # only take the X,Y axis data


    if nthist != 0:
        assert (design_matrix[:,-1] == ori_coord[:-nthist]).all()
        assert (design_matrix[:,1:-1] == filter_spikes(window_size, ori_spikes[nthist:])).all()

    else:
        assert (design_matrix[:,1:] == filter_spikes(window_size, ori_spikes)).all()

    assert (coord == ori_coord[nthist:]).all()



@pytest.mark.parametrize(
    "coord_axis, nthist, mode, window_size", 
    [
        ["y-axis", 0, "summed-current", 30],
        ["y-axis", 1, "summed-current", 30],
        ["y-axis", 0, "summed-past", 3],
        ["y-axis", 1, "summed-past", 3]
    ])
def test_SummedSpikesDataset(data_dir, coord_axis, nthist, mode, window_size):
    """Test SummedSpikesDataset."""
    def filter_spikes(window_size: int, design_spikes: NDArray) -> NDArray:
        """Filter spikes with the given kernel."""
        kernel = np.ones(window_size)

        def filtered(x: NDArray) -> NDArray:
            """Convovle with the given kernel."""
            return np.convolve(x, kernel, mode="same")

        return np.apply_along_axis(filtered, 0, design_spikes)

    design_matrix, coord = SummedSpikesDataset(data_dir, False, mode).load_all_data(coord_axis, nthist, window_size)

    coords_df = pd.read_csv(data_dir/'position.csv', index_col=0)
    ori_coord = coords_df.values[3:,1:3][:,0] if coord_axis == "x-axis" else coords_df.values[3:,1:3][:,1]# only take the X,Y axis data

    spikes_df = pd.read_csv(data_dir/'traces.csv', index_col=0)
    ori_spikes = spikes_df.values # only take the X,Y axis data

    if mode != "summed-past":
        if nthist != 0:
            assert (design_matrix[:,-1] == ori_coord[:-nthist]).all()
            assert (design_matrix[:,1:-1] == filter_spikes(window_size, ori_spikes[nthist:])).all()

        else:
            assert (design_matrix[:,1:] == filter_spikes(window_size, ori_spikes)).all()

        assert (coord == ori_coord[nthist:]).all()
    else:
        if nthist != 0:
            assert (design_matrix[:,-1] == ori_coord[:-nthist][(window_size//2):]).all()
            assert (design_matrix[:,1:-1] == filter_spikes(window_size, ori_spikes[nthist:])[:-(window_size//2)]).all()

        else:
            assert (design_matrix[:,1:] == filter_spikes(window_size, ori_spikes)[:-(window_size//2)]).all()

        assert (coord == ori_coord[nthist:][window_size//2:]).all()



if __name__ == "__main__":
    DATA_ROOT = Path('data/alldata/')
    coord_axis, nthist, mode, window_size = "y-axis", 0, "summed-past", 3
    test_SmoothedSpikesDataset(next(iter(DATA_ROOT.iterdir())), coord_axis, nthist, mode, window_size)