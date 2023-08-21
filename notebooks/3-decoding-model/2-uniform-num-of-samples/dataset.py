from typing import Optional, Tuple
from numpy.typing import NDArray

from dataclasses import dataclass
from sklearn.utils import resample
import numpy as np

from dataloader import UniformSegmentDataset

@dataclass
@dataclass
class ResampledDataset(UniformSegmentDataset):
    stand_y_classes: NDArray
    num_samples: int

    def __post_init__(self) -> None:
        super().__post_init__()

    def _resample_train_set(self, X, y):
        X_new, y_new = [], []
        for c in self.stand_y_classes:
            X_tmp, y_tmp = resample(X[y==c], y[y==c], n_samples=self.num_samples, random_state=self.random_state)
            X_new.append(X_tmp)
            y_new.append(y_tmp)
        return np.vstack(X_new), np.hstack(y_new)
    
    def load_all_data(self, window_size: int, K: int, train_ratio: Optional[float] = None) -> Tuple:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = super().load_all_data(window_size, K, train_ratio)
        self.X_train, self.y_train = self._resample_train_set(self.X_train, self.y_train)
        return (self.X_train, self.y_train), (self.X_test, self.y_test)