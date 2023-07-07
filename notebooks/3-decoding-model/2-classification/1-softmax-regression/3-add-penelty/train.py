from typing import Tuple

from tqdm import tqdm
import pickle
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

from dataloader.dataset import BaseDataset
from param import *
from utils.util import downsample

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

    Train with downsampling.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = Dataset(data_dir, False, False)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio)

        results_all = []
        for c in ParamTrain().C_range:
            model =  LogisticRegression(
                    multi_class='multinomial', 
                    solver='newton-cg',
                    penalty= ParamTrain().penalty,
                    C=c
                    ) #SoftmaxRegression() #sm.MNLogit(y_train.ravel(), X_train)

            # fit
            # result = model.fit(method="ncg")
            result = model.fit(X_train, y_train)

            # y_pred = model.predict(params=result.params ,exog = X_test) #model.predict(X_test, beta)
            y_pred = model.predict(X_test)

            results = {
                "C": c,
                "estimator": model,
                "y_test": y_test,
                "y_pred": y_pred #np.array([y+1 for y in np.argmax(y_pred, axis=1)])
            }
            results_all.append(results)
        with open(ParamDir().output_dir/data_name/(f"sm_train_with_penalty.pickle"),"wb") as f:
            pickle.dump(results_all, f)

if __name__ == "__main__":
    main()