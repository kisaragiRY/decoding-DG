from typing import Tuple

import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA

from modules.dataloader import UniformSegmentDataset
from param import *
from modules.utils.util import segment, downsample, segment_with_threshold, get_segment_data

@dataclass
class ConcatDataset(UniformSegmentDataset):
    def __post_init__(self):
        super().__post_init__()

    def load_all_data(self, window_size, K, train_ratio : int) -> Tuple:
        """Load design matrix and corresponding response(coordinate).
        
        Parameter
        ------------
        window_size : int
            smoothing window size.
        """
        (self.X_train, self.y_train), (self.X_test, self.y_test) = super().load_all_data(window_size, K, train_ratio)

        X_train_concat = np.concatenate(self.X_train, 0)
        y_train_concat = np.repeat(self.y_train, self.X_train.shape[1])

        X_test_concat = np.concatenate(self.X_test, 0)
        y_test_concat = np.repeat(self.y_test, self.X_test.shape[1])

        return (X_train_concat, y_train_concat), (X_test_concat, y_test_concat)