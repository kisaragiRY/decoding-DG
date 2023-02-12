from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tqdm import tqdm
import pickle
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from dataloader.dataset import BaseDataset
from param import *
from util import downsample
# from decoder import SoftmaxRegression


@dataclass
class Dataset(BaseDataset):
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

        # --- smooth data
        self.X_train = self._filter_spikes(window_size, self.X_train) 
        self.X_test = self._filter_spikes(window_size, self.X_test)

        # -- normaliza data
        self.X_train = (self.X_train - self.X_train.mean(axis=0))/self.X_train.std(axis=0)
        self.X_test = (self.X_test - self.X_test.mean(axis=0))/self.X_train.std(axis=0)

        # -- upsample
        # oversample = SMOTE()
        # self.X_train, self.y_train = oversample.fit_resample(self.X_train, self.y_train)
        # self.X_test, self.y_test = oversample.fit_resample(self.X_test, self.y_test)

        # -- downsample
        self.X_train, self.y_train = downsample(self.X_train, self.y_train)
        self.X_test, self.y_test = downsample(self.X_test, self.y_test)

        # --- add offset(intercept)
        # self.X_train = np.hstack((np.ones((len(self.X_train),1)), self.X_train))
        # self.X_test = np.hstack((np.ones((len(self.X_test),1)), self.X_test))

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

def main():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list[[1, 2]]):
        data_name = str(data_dir).split('/')[-1]
        
        dataset = Dataset(data_dir, False, False)

        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio)

        sm = LogisticRegression(multi_class='multinomial', solver='lbfgs') #SoftmaxRegression()
        losses, beta = sm.fit(X_train, y_train, lr = ParamTrain().lr, max_iter = ParamTrain().max_iter)

        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/(f"sm_training_firing_rate.pickle"),"wb") as f:
            pickle.dump((losses, beta),f)
    
if __name__ == "__main__":
    main()

