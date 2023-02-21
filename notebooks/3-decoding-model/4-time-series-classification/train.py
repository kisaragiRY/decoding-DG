from typing import Tuple

import pandas as pd
from tqdm import tqdm
import pickle
from sktime.classification.kernel_based import RocketClassifier
import statsmodels.api as sm

from dataloader.dataset import BaseDataset
from param import *
from util import segment

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

        # --- remove inactive neurons
        active_neurons = self.X_train.sum(axis=0)>0
        self.X_train = self.X_train[:, active_neurons]
        self.X_test = self.X_test[:, active_neurons]

        # --- smooth data
        self.X_train = self._filter_spikes(window_size, self.X_train) 
        self.X_test = self._filter_spikes(window_size, self.X_test)

        # --- segment data
        segment_ind = segment(self.y_train) # get the segmentation indices
        y_new = np.append(self.y_train[0], self.y_train[segment_ind]) # segment y
        X_seg = np.split(self.X_train, segment_ind) # segment X
        max_len = max([len(X) for X in X_seg])
        n_neurons = X_seg[0].shape[1]
        X_seg_new, y_new_train = [], []
        for _id, X in enumerate(X_seg):
            if len(X) > 3: # the instance time points need to be more than 3 bins
                y_new_train.append(str(y_new[_id]))
                # X_seg_new.append(X) # unequal length
                X_seg_new.append(np.vstack((X, np.zeros((max_len - len(X), n_neurons))))) # set to equal length with zeros

        # filter the neuron: delete the neurons where the activity is zero across instances
        neurons_to_use = np.vstack(X_seg_new).sum(axis=0)>0
        X_seg_new = [X[:, neurons_to_use ] for X in X_seg_new]

        self.y_train = np.array(y_new_train)
        self.X_train = pd.DataFrame([[pd.Series(i) for i in X.T] for X in X_seg_new])

        # ---- test set
        segment_ind = segment(self.y_test)

        y_new = np.append(self.y_test[0], self.y_test[segment_ind])

        X_seg = np.split(self.X_test, segment_ind)
        X_seg_new, y_new_test = [], []
        for _id, X in enumerate(X_seg):
            if (len(X) <= max_len) and (len(X) > 3):
                y_new_test.append(str(y_new[_id]))
                # X_seg_new.append(X) # unequal length
                X_seg_new.append(np.vstack((X, np.zeros((max_len - len(X), n_neurons))))) # set to equal length with zeros

        # filter the neuron: delete the neurons where the activity is zero across instances
        X_seg_new = [X[:, neurons_to_use ] for X in X_seg_new]

        self.y_test = np.array(y_new_test)
        self.X_test = pd.DataFrame([[pd.Series(i) for i in X.T] for X in X_seg_new])


        # --- add offset(intercept)
        # self.X_train = np.hstack((np.ones((len(self.X_train),1)), self.X_train))
        # self.X_test = np.hstack((np.ones((len(self.X_test),1)), self.X_test))

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

def main():
    """The training script.

    Train with downsampling.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = Dataset(data_dir, ParamData().mobility, False)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio)

        model =  RocketClassifier(num_kernels=2000)

        # fit
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results = {
            "estimator": model,
            "y_test": y_test,
            "y_pred": y_pred #np.array([y+1 for y in np.argmax(y_pred, axis=1)])
        }
        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/(f"tsc_train_rocket.pickle"),"wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main()