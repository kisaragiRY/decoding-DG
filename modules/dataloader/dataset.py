from numpy.typing import NDArray
from typing import Tuple, Optional, Union

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from util import gauss1d, cal_velocity, bin_pos, segment_with_threshold, get_segment_data, downsample

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
        the value can be either False, 'behavior shuffling' or 'events shuffling'.
    """
    data_dir : Path
    mobility : Union[float, bool]
    shuffle_method : Union[str, bool]
    random_state: Union[int, bool] 

    def __post_init__(self) -> None:
        """Post precessing."""
        self.coords_xy, self.spikes = self._load_raw_data()
        if self.mobility:
            vel = cal_velocity(self.coords_xy)
            self.coords_xy = self.coords_xy[vel > self.mobility]
            self.spikes = self.spikes[vel > self.mobility]

        if self.shuffle_method:
            if self.shuffle_method not in ['behavior shuffling', 'events shuffling']:
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
        else:
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
    coordinates and gassian kernel smoothed spikes.
    """  
    coord_axis : str
    def _filter_spikes(self, window_size: int, X: NDArray) -> NDArray:
        """Filter spikes with the given kernel."""
        kernel = gauss1d(np.linspace(-3, 3, window_size))

        def filtered(x: NDArray) -> NDArray:
            """Convovle with the given kernel."""
            return np.convolve(x, kernel, mode="same")

        return np.apply_along_axis(filtered, 0, X)

    def load_all_data(self, window_size : int, train_ratio: float) -> Tuple[NDArray, NDArray]:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        coord_axis : str
            Specify which axis to use. 
            The coord_axis can either be 'x-axis' or 'y-axis'.
        nthist: int
            Which history bin to use ahead of current bin.
        """

        # --- split data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.split_data(self.X, self.y, train_ratio)

        # --- smooth data
        self.X_train = self._filter_spikes(window_size, self.X_train) 
        self.X_test = self._filter_spikes(window_size, self.X_test)

        # --- add offset(intercept)
        self.X_train = np.hstack((np.ones((len(self.X_train),1)), self.X_train))
        self.X_test = np.hstack((np.ones((len(self.X_test),1)), self.X_test))

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

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

@dataclass
class UniformSegmentDataset(BaseDataset):
    """A balanced dataset where segmenting is based on a given threshold so that
    each segment is with the same length.
    """
    def __post_init__(self):
        super().__post_init__()
        self.y = self._discretize_coords()
        self.X = self.spikes
    
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
        if train_ratio:
            # --- split data 
            (self.X_train, self.y_train), (self.X_test, self.y_test) = self.split_data(self.X, self.y, train_ratio)

            # --- remove inactive neurons
            active_neurons = self.X_train.sum(axis=0)>0
            self.X_train = self.X_train[:, active_neurons]
            self.X_test = self.X_test[:, active_neurons]

            # --- segment data while smoothing
            X_train_new, self.y_train = self._get_segment_data((self.X_train, self.y_train), window_size, K)
            X_test_new, self.y_test = self._get_segment_data((self.X_test, self.y_test), window_size, K)

            # -- downsample
            self.X_train, self.y_train = downsample(X_train_new, self.y_train, self.random_state)
            self.X_test, self.y_test = downsample(X_test_new, self.y_test, self.random_state)

            return (self.X_train, self.y_train), (self.X_test, self.y_test)

        else:
            self.X_train, self.y_train = self.X, self.y

            # --- remove inactive neurons
            active_neurons = self.X_train.sum(axis=0)>0
            self.X_train = self.X_train[:, active_neurons]

            # --- segment data while smoothing
            X_train_new, self.y_train = self._get_segment_data((self.X_train, self.y_train), window_size, K)

            # -- downsample
            self.X_train, self.y_train = downsample(X_train_new, self.y_train, self.random_state)

            return self.X_train, self.y_train