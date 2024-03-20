from numpy.typing import NDArray
from typing import Tuple, Optional, Union, Literal

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.util import *

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
    mobility : Optional[float]
        whethe to only use the data when the mouse is moving.
        if given, must specify the threshold for identifying immobility.
    shuffle_method : Optional[str]
        whether to shuffle the data, and if does, specify the method.
        the value can be either False, 'behavior shuffling', 'events shuffling' or "segment label shuffling".
    """
    data_dir : Path
    mobility : Union[float, bool]
    shuffle_method : Union[Literal['behavior shuffling', 'events shuffling', 'segment label shuffling'], bool]
    random_state: Union[int, bool] 

    def __post_init__(self) -> None:
        """Post precessing."""
        self.coords_xy, self.spikes = self._load_raw_data()
        if self.mobility:
            self.vel_base = cal_velocity(self.coords_xy)
            self.coords_xy = self.coords_xy[self.vel_base > self.mobility]
            self.spikes = self.spikes[self.vel_base > self.mobility]

        if self.shuffle_method:
            if self.shuffle_method not in ['behavior shuffling', 'events shuffling', 'segment label shuffling']:
                raise ValueError("Please specify a valid shuffle method. It can either be 'behavior shuffling' or 'events shuffling'.")
            else:
                self._shuffle()

    def _load_raw_data(self) -> Tuple[NDArray, NDArray]:
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
        if self.random_state:
            np.random.seed(self.random_state)
            
        if self.shuffle_method == 'behavior shuffling':
            # --- 1. flip in time
            self.coords_xy = self.coords_xy[::-1]
            # --- 2. shift a random amount
            random_num = np.random.randint(1, len(self.coords_xy))
            self.coords_xy = np.roll(self.coords_xy, random_num)
        elif self.shuffle_method == 'events shuffling':
            for row in self.spikes:
                # shuffle the row when there are spikes
                if np.sum(row) > 0:
                    np.random.shuffle(row)
    
    def split_data(self, X: NDArray, y: NDArray, train_ratio: float) -> Tuple:
        """Split the data into train set and test set.
        """
        train_size = int( len(X) * train_ratio )
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        return (X_train, y_train), (X_test, y_test)
    
    def _filter_spikes(self, window_size: int, X: NDArray) -> NDArray:
        """Filter spikes with a gaussian kernel."""
        kernel = gauss1d(np.linspace(-3, 3, window_size))

        def filtered(x: NDArray) -> NDArray:
            """Convovle with the given kernel."""
            return np.convolve(x, kernel, mode="same")

        return np.apply_along_axis(filtered, 0, X)
    
    def _discretize_coords(self):
        """Discretize the coordinates.
        """
        return bin_pos(self.coords_xy)

@dataclass
class UniformSegmentDataset(BaseDataset):
    """A balanced dataset where segmenting is based on a given threshold so that
    each segment is with the same length.
    """
    def __post_init__(self):
        super().__post_init__()
    
    def _get_segment_data(self, data: Tuple[NDArray, NDArray],window_size: int, K: int):
        """Get segmented data.
        """
        X, y = data
        segment_ind = segment_with_threshold(y, K) # get the segmentation indices
        X_new, y_new = get_segment_data(segment_ind, K, window_size, X, y)

        return X_new, y_new
    
    def load_all_data(self, window_size : int, K: int, train_ratio: Optional[float] = None) -> Tuple:
        """Load train and test set if train_ratio is set.
        
        Parameter
        ------------
        window_size : int
            smoothing window size.
        K: int
            segment length threshold.
        train_ratio : Optional[float] = None
            the training set ration, in the range of 0 and 1.
        """

        # --- split data 
        (self.X_train_base, self.y_train_base), (self.X_test_base, self.y_test_base) = self.split_data(self.spikes, self._discretize_coords(), train_ratio)

        # --- remove inactive neurons
        self.active_neurons = self.X_train_base.sum(axis=0)>0
        self.X_train_active = self.X_train_base[:, self.active_neurons]
        self.X_test_active = self.X_test_base[:, self.active_neurons]

        # --- segment data while smoothing
        self.X_train_seg, self.y_train_seg = self._get_segment_data((self.X_train_active, self.y_train_base), window_size, K)
        self.X_test_seg, self.y_test_seg = self._get_segment_data((self.X_test_active, self.y_test_base), window_size, K)

        # -- downsample
        self.X_train, self.y_train = downsample(self.X_train_seg, self.y_train_seg, self.random_state)
        self.X_test, self.y_test = downsample(self.X_test_seg, self.y_test_seg, self.random_state)
        
        # -- shuffle
        if self.shuffle_method == 'segment label shuffling':
            np.random.shuffle(self.y_train)
            np.random.shuffle(self.y_test)

        return (self.X_train, self.y_train), (self.X_test, self.y_test)


