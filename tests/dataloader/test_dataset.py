import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from modules.dataloader import PastCoordDataset, SpikesCoordDataset

@pytest.fixture
def data_dir():
    DATA_ROOT = Path('data/alldata/')
    data_dir = next(iter(DATA_ROOT.iterdir()))
    return data_dir

@pytest.mark.parametrize("coord_axis, nthist", [["x-axis", 1], ["y-axis", 3]])
def test_PastCoordDataset(data_dir, coord_axis, nthist):
    design_matrix, coord = PastCoordDataset(data_dir).load_all_data(coord_axis, nthist)

    coords_df = pd.read_csv(data_dir/'position.csv', index_col=0)
    ori_coord = coords_df.values[3:,1:3][:,0] if coord_axis == "x-axis" else coords_df.values[3:,1:3][:,1]# only take the X,Y axis data

    assert (ori_coord[nthist:].ravel() == coord.ravel()).all()
    assert (design_matrix[:,-1] == ori_coord[:-nthist]).all()

@pytest.mark.parametrize("coord_axis, nthist", [["x-axis", 0], ["x-axis", 1], ["y-axis", 2]])
def test_SpikesCoordDataset(data_dir, coord_axis, nthist):
    design_matrix, coord = SpikesCoordDataset(data_dir).load_all_data(coord_axis, nthist)

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