from numpy.typing import NDArray
from typing import Tuple, Union

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from util import estimate_firing_rate

def _is_valid_axis(coord_axis: str) -> bool:
    """Check whether the axis is valid.
    
    The axis can either be 'x-axis' or 'y-axis'.
    """
    if coord_axis in ['x-axis','y-axis']:
        return True
    return False

@dataclass
class BaseDataset:
    """Base dataset that loads spikes and coordinates data.

    Parameters
    ---------
    datadir : Path
        the path to a mouse's data
    shuffle_method : Union[bool, str]
        whether to shuffle the data, and if does, specify the method.
        the value can be either False, 'behavior shuffling' or 'events shuffling'.
    """
    data_dir : Path
    shuffle_method : Union[bool, str]

    def __post_init__(self) -> None:
        """Post precessing."""
        self.coords_xy, self.spikes = self._load_data()
        if self.shuffle_method:
            if self.shuffle_method not in ['behavior shuffling', 'events shuffling']:
                raise ValueError("Please specify a valid shuffle method. It can either be 'behavior shuffling' or 'events shuffling'.")
            else:
                self._shuffle()

    def _load_data(self) -> Tuple[NDArray, NDArray]:
        """Load coordinates and spike data."""
        coords_df = pd.read_csv(self.data_dir/'position.csv',index_col=0)
        coords = coords_df.values[3:,1:3] # only take the X,Y axis data

        spikes_df = pd.read_csv(self.data_dir/'traces.csv',index_col=0)
        spikes = spikes_df.values

        # make sure spike and postion data have the same length
        n_bins = min(len(coords),len(spikes))
        coords = coords[:n_bins]
        spikes = spikes[:n_bins]

        return coords, spikes
    
    def _shuffle(self) -> None:
        """Shuffle the data.
        
        Based on two methods:'behavior shuffling' and 'events shuffling'.
        Details see method in reference: https://pubmed.ncbi.nlm.nih.gov/32521223/
        """
        if self.shuffle_method == 'behavior shuffling':
            # --- 1. flip in time
            self.shuffled_coords_xy = self.coords_xy[::-1]
            # --- 2. shift a random amount
            random_num = np.random.randint(1, len(self.coords_xy))
            self.shuffled_coords_xy = np.roll(self.coords_xy, random_num)
        else:
            self.shuffle_spikes = self.spikes
            for row in self.shuffle_spikes:
                # shuffle the row when there are spikes
                if np.sum(row) > 0:
                    np.random.shuffle(row)
    

@dataclass
class SpikesCoordDataset(BaseDataset):
    """Dataset that includes one mouse's spikes and coordinates.
    
    This dataset is for regression model that incorporates past 
    coordinates as one of the features in design matrix.
    """
    def design_matrix(self, nthist: int) -> NDArray:
        """Make design matrix for decoder with past corrdinates.

        Parameter:
        ----------
        spikes: NDArray
            that has neurons's spikes count data.
        coordinates: NDArray
            x or y coordinate data
        nthist: int
            num of time bins for spikes history, default=1
        """
        n_time_bins, n_neurons = self.spikes.shape
        if nthist !=0:
            design_m = np.zeros((n_time_bins - nthist, n_neurons+1))
            design_m[:,:-1] = self.spikes[nthist:]
            design_m[:,-1] = self.coord[:-nthist]
        else:
            design_m = self.spikes

        design_mat_all_offset = np.hstack((np.ones((n_time_bins-nthist,1)), design_m))
        return design_mat_all_offset

    def load_all_data(self, coord_axis: str, nthist: int) -> Tuple[NDArray, NDArray]:
        """Load design matrix and corresponding response(coordinate)."""
        self.axis = 0 if coord_axis == "x-axis" else 1
        if not _is_valid_axis(coord_axis):
            raise ValueError("The coord_axis can either be 'x-axis' or 'y-axis'.")
        self.coord = self.coords_xy[:,self.axis]
        return self.design_matrix(nthist), self.coord[nthist:]

@dataclass
class PastCoordDataset(BaseDataset):
    """Dataset that includes one mouse's spikes and coordinates.
    
    This dataset is for regression model that incorporates past 
    coordinates as one of the features in design matrix.

    """

    def design_matrix(self, nthist: int) -> NDArray:
        """Make design matrix for decoder with past corrdinates.

        Parameter:
        ----------
        spikes: NDArray
            that has neurons's spikes count data.
        coordinates: NDArray
            x or y coordinate data
        nthist: int
            num of time bins for spikes history
        """
        if nthist !=0:
            design_m = self.coord[:-nthist].reshape(-1,1)
        else:
            raise ValueError("nthist must be larger than 0.")

        design_mat_all_offset = np.hstack((np.ones((len(self.coord)-nthist,1)), design_m))
        return design_mat_all_offset

    def load_all_data(self, coord_axis : str, nthist : int) -> Tuple[NDArray, NDArray]:
        """Load design matrix and corresponding response(coordinate)."""
        if not _is_valid_axis(coord_axis):
            raise ValueError("The coord_axis can either be 'x-axis' or 'y-axis'.")

        self.axis = 0 if coord_axis == "x-axis" else 1
        self.coord = self.coords_xy[:, self.axis]
        return self.design_matrix(nthist), self.coord[nthist:]

@dataclass
class SmoothedSpikesDataset(BaseDataset):
    """Dataset that includes one mouse's spikes and coordinates.
    
    This dataset is for regression model that incorporates past 
    coordinates(optional) and gassian kernel smoothed spikes.
    """  
    def load_all_data(self, coord_axis : str, nthist : int, window_size : int, sigma:float = .2) -> Tuple[NDArray, NDArray]:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        coord_axis : str
            Specify which axis to use. 
            The coord_axis can either be 'x-axis' or 'y-axis'.
        nthist: int
            Which history bin to use ahead of current bin.
        """
        if not _is_valid_axis(coord_axis):
            raise ValueError("Invalid axis name. The coord_axis can either be 'x-axis' or 'y-axis'.")

        self.axis = 0 if coord_axis == "x-axis" else 1
        self.coord = self.coords_xy[:, self.axis]

        if nthist != 0:
            design_m[:,:-1] = estimate_firing_rate(window_size, self.spikes[nthist:], sigma)
            design_m[:,-1] = self.coord[:-nthist]
        else:
            design_m = estimate_firing_rate(self.spikes, window_size, sigma) 

        design_mat_all_offset = np.hstack((np.ones((len(design_m),1)), design_m))

        return design_mat_all_offset, self.coord[nthist:]

@dataclass
class SummedSpikesDataset(BaseDataset):
    """Dataset that includes one mouse's spikes and coordinates.
    
    This dataset is for regression model that incorporates past 
    coordinates and history or future spikes.

    Parameters
    ---------
    mode : str
        In which way to incorparate the history spikes data depending on the window size.
        "summed-past" : sum up the past spikes, date back from the current bin.
        "summed-current" : sum up the spikes centered by the current bin.
    """
    mode : str

    def __post_init__(self) -> None:
        """Post precessing."""
        if self.mode not in ["summed-past", "summed-current"]:
            raise ValueError("Invalid mode name. The coord_mode can either be 'summed-past' or 'summed-current'.")
        self.coords_xy, self.spikes = self._load_data()
    
    def filter_spikes(self, window_size: int, design_spikes: NDArray) -> NDArray:
        """Filter spikes with the given kernel."""
        kernel = np.ones(window_size)

        def filtered(x: NDArray) -> NDArray:
            """Convovle with the given kernel."""
            return np.convolve(x, kernel, mode="same")

        return np.apply_along_axis(filtered, 0, design_spikes)

    def load_all_data(self, coord_axis : str, nthist : int, window_size : int) -> Tuple[NDArray, NDArray]:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        coord_axis : str
            Specify which axis to use. 
            The coord_axis can either be 'x-axis' or 'y-axis'.
        nthist: int
            Which history bin to use ahead of current bin.
        """
        if not _is_valid_axis(coord_axis):
            raise ValueError("Invalid axis name. The coord_axis can either be 'x-axis' or 'y-axis'.")

        self.axis = 0 if coord_axis == "x-axis" else 1
        self.coord = self.coords_xy[:, self.axis]

        n_time_bins, n_neurons = self.spikes.shape
        if nthist != 0:
            design_m = np.zeros((n_time_bins - nthist, n_neurons+1))
            design_m[:,:-1] = self.filter_spikes(window_size, self.spikes[nthist:]) 
            design_m[:,-1] = self.coord[:-nthist]
            if self.mode == "summed-past" and window_size!=1:
                design_m_spikes = design_m[:,:-1][:-(window_size//2)]
                design_m_coord = design_m[:,-1][window_size//2:]
                design_m = np.hstack((design_m_spikes, design_m_coord.reshape(-1,1)))
                self.coord = self.coord[window_size//2:]
        else:
            design_m = self.filter_spikes(window_size, self.spikes) 
            if self.mode == "summed-past" and window_size!=1:
                design_m = design_m[:-(window_size//2)]
                self.coord = self.coord[window_size//2:]


        design_mat_all_offset = np.hstack((np.ones((len(design_m),1)), design_m))

        return design_mat_all_offset, self.coord[nthist:]


