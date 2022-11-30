from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

def _is_valid_axis(coord_axis: str) -> bool:
    """Check whether the axis is valid.
    
    The axis can either be 'x-axis' or 'y-axis'.
    """
    if coord_axis in ['x-axis','y-axis']:
        return True
    return False

@dataclass
class SpikesPastCoordDataset:
    """Dataset that includes one mouse's spikes and coordinates.
    
    This dataset is for regression model that incorporates past 
    coordinates as one of the features in design matrix.

    Parameters
    ---------
    datadir : Path
        the path to a mouse's data
    coord_axis : str
        which axis to use
    nthist : int
        the number of history bins
    
    """
    data_dir : Path
    coord_axis : str
    nthist : int

    def __post_init__(self) -> None:
        """Post precessing."""
        if not _is_valid_axis(self.coord_axis):
            raise ValueError("The coord_axis can either be 'x-axis' or 'y-axis'.")
        self.axis = 0 if self.coord_axis == "x-axis" else 1
        self.data = self.load_all_data()

    @property
    def design_matrix(self) -> np.array:
        """Make design matrix for decoder with past corrdinates.

        Parameter:
        ----------
        spikes: np.array
            that has neurons's spikes count data.
        coordinates: np.array
            x or y coordinate data
        nthist: int
            num of time bins for spikes history, default=1
        """
        n_time_bins, n_neurons = self.spikes.shape
        design_m = np.zeros((n_time_bins - self.nthist, n_neurons+1))
        if self.nthist !=0:
            design_m[:,:-1] = self.spikes[self.nthist:]
            design_m[:,-1] = self.coord[:-self.nthist]
        else:
            design_m = self.spikes

        design_mat_all_offset = np.hstack((np.ones((n_time_bins-self.nthist,1)), design_m))
        return design_mat_all_offset

    def load_all_data(self) -> Tuple[np.array, np.array]:
        """Load design matrix and corresponding response(coordinate)."""
        coords_xy, self.spikes = self._load_data()
        self.coord = coords_xy[:,self.axis]
        return self.design_matrix, self.coord[self.nthist:]

    def _load_data(self) -> Tuple[np.array, np.array]:
        """Load coordinates and spike data."""
        coords_df = pd.read_excel(self.data_dir/'position.xlsx')
        coords=coords_df.values[3:,1:3] # only take the X,Y axis data

        spikes_df = pd.read_excel(self.data_dir/'traces.xlsx',index_col=0)
        spikes = spikes_df.values

        # make sure spike and postion data have the same length
        n_bins = min(len(coords),len(spikes))
        coords = coords[:n_bins]
        spikes = spikes[:n_bins]

        return coords,spikes


@dataclass
class PastCoordDataset:
    """Dataset that includes one mouse's spikes and coordinates.
    
    This dataset is for regression model that incorporates past 
    coordinates as one of the features in design matrix.

    Parameters
    ---------
    datadir : Path
        the path to a mouse's data
    coord_axis : str
        which axis to use
    nthist : int
        the number of history bins
    
    """
    data_dir : Path
    coord_axis : str
    nthist : int

    def __post_init__(self) -> None:
        """Post precessing."""
        if not _is_valid_axis(self.coord_axis):
            raise ValueError("The coord_axis can either be 'x-axis' or 'y-axis'.")
        self.axis = 0 if self.coord_axis == "x-axis" else 1
        self.data = self.load_all_data()

    @property
    def design_matrix(self) -> np.array:
        """Make design matrix for decoder with past corrdinates.

        Parameter:
        ----------
        spikes: np.array
            that has neurons's spikes count data.
        coordinates: np.array
            x or y coordinate data
        nthist: int
            num of time bins for spikes history
        """
        n_time_bins, _ = self.spikes.shape
        if self.nthist !=0:
            design_m = self.coord[:-self.nthist]
        else:
            raise ValueError("nthist must be larger than 0.")

        design_mat_all_offset = np.hstack((np.ones((n_time_bins-self.nthist,1)), design_m))
        return design_mat_all_offset

    def load_all_data(self) -> Tuple[np.array, np.array]:
        """Load design matrix and corresponding response(coordinate)."""
        coords_xy, self.spikes = self._load_data()
        self.coord = coords_xy[:,self.axis]
        return self.design_matrix, self.coord[self.nthist:]

    @cached_property
    def _load_data(self) -> Tuple[np.array, np.array]:
        """Load coordinates and spike data."""
        coords_df = pd.read_excel(self.data_dir/'position.xlsx')
        coords=coords_df.values[3:,1:3] # only take the X,Y axis data

        spikes_df = pd.read_excel(self.data_dir/'traces.xlsx',index_col=0)
        spikes = spikes_df.values

        # make sure spike and postion data have the same length
        n_bins = min(len(coords),len(spikes))
        coords = coords[:n_bins]
        spikes = spikes[:n_bins]

        return coords,spikes
    



