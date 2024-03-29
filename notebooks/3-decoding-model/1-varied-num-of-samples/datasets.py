from typing import Tuple

import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA

from modules.dataloader.dataset import BaseDataset
from param import *
from modules.utils.util import segment, downsample, segment_with_threshold, get_segment_data

@dataclass
class DownsampleDataset(BaseDataset):
    def __post_init__(self):
        super().__post_init__()
        self.y_train = self._discretize_coords()
        self.X_train = self.spikes

        # --- remove inactive neurons
        active_neurons = self.X_train.sum(axis=0)>0
        self.X_train = self.X_train[:, active_neurons]

    def load_all_data(self, window_size : int) -> Tuple:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        window_size : int
            smoothing window size.
        """
        # --- smooth data
        self.X_train = self._filter_spikes(window_size, self.X_train) 

        # -- downsample
        self.X_train, self.y_train = downsample(self.X_train, self.y_train)

        return self.X_train, self.y_train
    
@dataclass
class SegmentDataset(BaseDataset):
    def __post_init__(self):
        super().__post_init__()
        self.y = self._discretize_coords()

    def load_all_data(self, window_size : int, train_ratio: float) -> Tuple:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        window_size : int
            smoothing window size.
        train_ratio: float
            train set ratio
        """
        self.y = self._discretize_coords()
        self.X = self.spikes

        # --- split data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.split_data(self.X, self.y, train_ratio)

        # --- remove inactive neurons
        active_neurons = self.X_train.sum(axis=0)>0
        self.X_train = self.X_train[:, active_neurons]
        self.X_test = self.X_test[:, active_neurons]

        # # --- smooth data
        # self.X_train = self._filter_spikes(window_size, self.X_train) 
        # self.X_test = self._filter_spikes(window_size, self.X_test)

        # --- segment data
        segment_ind = segment(self.y_train) # get the segmentation indices
        y_new = np.append(self.y_train[0], self.y_train[segment_ind]) # segment y
        X_seg = np.split(self.X_train, segment_ind) # segment X
        max_len = max([len(X) for X in X_seg])
        n_neurons = X_seg[0].shape[1]
        X_seg_new, y_new_train = [], []
        for _id, X in enumerate(X_seg):
            if len(X) > 3: # the instance time points need to be more than 3 bins
                X = self._filter_spikes(window_size, X) # smooth the interval
                y_new_train.append(str(y_new[_id]))
                # X_seg_new.append(X) # unequal length
                X_seg_new.append(np.vstack((X, np.zeros((max_len - len(X), n_neurons)))).T) # set to equal length with zeros


        # filter the neuron: delete the neurons where the activity is zero across instances
        neurons_to_use = np.vstack(X_seg_new).sum(axis=0)>0
        X_seg_new = [X[:, neurons_to_use ] for X in X_seg_new]

        self.y_train = np.array(y_new_train)
        self.X_train = np.array(X_seg_new)#pd.DataFrame([[pd.Series(i) for i in X.T] for X in X_seg_new])

        # test set
        segment_ind = segment(self.y_test)

        y_new = np.append(self.y_test[0], self.y_test[segment_ind])

        X_seg = np.split(self.X_test, segment_ind)
        X_seg_new, y_new_test = [], []
        for _id, X in enumerate(X_seg):
            if (len(X) <= max_len) and (len(X) > 3):
                X = self._filter_spikes(window_size, X) 
                y_new_test.append(str(y_new[_id]))
                # X_seg_new.append(X) # unequal length
                X_seg_new.append(np.vstack((X, np.zeros((max_len - len(X), n_neurons)))).T) # set to equal length with zeros

        # filter the neuron: delete the neurons where the activity is zero across instances
        X_seg_new = [X[:, neurons_to_use ] for X in X_seg_new]

        self.y_test = np.array(y_new_test)
        self.X_test = np.array(X_seg_new)#pd.DataFrame([[pd.Series(i) for i in X.T] for X in X_seg_new])


        return (self.X_train, self.y_train), (self.X_test, self.y_test)
    
@dataclass
class BalancedSegmentDataset(SegmentDataset):
    """Balance Dataset.
    """
    def __post_init__(self):
        super().__post_init__()

    def load_all_data(self, window_size: int, train_ratio: float) -> Tuple:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = super().load_all_data(window_size, train_ratio)
        # -- downsample
        self.X_train, self.y_train = downsample(self.X_train, self.y_train)
        self.X_test, self.y_test = downsample(self.X_test, self.y_test)

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

@dataclass
class ThresholdSegmentDataset(BaseDataset):
    """Segmenting based on a given threshold balanced dataset.
    """
    def __post_init__(self):
        super().__post_init__()
        self.y = self._discretize_coords()
        self.X = self.spikes
    
    def load_all_data(self, window_size : int, train_ratio: float, K: int) -> Tuple:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        window_size : int
            smoothing window size.
        K: int
            segment length threshold.
        """
        # --- split data 
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.split_data(self.X, self.y, train_ratio)

        # --- remove inactive neurons
        active_neurons = self.X_train.sum(axis=0)>0
        self.X_train = self.X_train[:, active_neurons]
        self.X_test = self.X_test[:, active_neurons]
        
        # --- segment data while smoothing
        # train set
        segment_ind = segment_with_threshold(self.y_train, K) # get the segmentation indices
        X_train_new, self.y_train = get_segment_data(segment_ind, K, window_size, self.X_train, self.y_train)
        # test set
        segment_ind = segment_with_threshold(self.y_test, K) # get the segmentation indices
        X_test_new, self.y_test = get_segment_data(segment_ind, K, window_size, self.X_test, self.y_test)

        # # filter the neuron: delete the neurons where the activity is zero across instances
        # neurons_to_use = np.vstack(X_train_new).sum(axis=0)>0
        # self.X_train = np.array([X[:, neurons_to_use ] for X in X_train_new])
        # self.X_test = np.array([X[:, neurons_to_use ] for X in X_test_new])

        # -- downsample
        self.X_train, self.y_train = downsample(X_train_new, self.y_train)
        self.X_test, self.y_test = downsample(X_test_new, self.y_test)


        return (self.X_train, self.y_train), (self.X_test, self.y_test)

@dataclass
class DimRedDataset(BaseDataset):
    """Dimension reduction balanced dataset.
    """
    def __post_init__(self):
        super().__post_init__()
        self.y_train = self._discretize_coords()
        self.X_train = self.spikes
    
    def load_all_data(self, window_size: int, reduction_method: str) -> Tuple:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        window_size: int
            smoothing window size.
        reduction_method: str
            the reduction method to reduce dimesions of the data.
        """
        # --- remove inactive neurons
        active_neurons = self.X_train.sum(axis=0)>0
        self.X_train = self.X_train[:, active_neurons]
        
        # --- smooth data
        self.X_train = self._filter_spikes(window_size, self.X_train) 

        if reduction_method == "LEM":
            self.X_train = SpectralEmbedding(n_components=10, 
                                             n_neighbors=int(.005*len(self.X_train))).fit_transform(self.X_train)
        elif reduction_method == "PCA":
            self.X_train = PCA(n_components=10).fit_transform(self.X_train)

        # -- downsample
        self.X_train, self.y_train = downsample(self.X_train, self.y_train)

        return self.X_train, self.y_train
