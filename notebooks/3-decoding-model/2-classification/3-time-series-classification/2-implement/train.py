from typing import Tuple

import pandas as pd
from tqdm import tqdm
import pickle
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import statsmodels.api as sm

from dataloader.dataset import BaseDataset
from param import *
from util import segment, downsample

def rocket_trainer():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = Dataset(data_dir, ParamData().mobility, False)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio)

        model =  RocketClassifier(
            num_kernels= ParamaRocketTrain().num_kernels,
            rocket_transform = "rocket",
            use_multivariate = "yes")

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

def rocket_trainer_balanced():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = BalancedDataset(data_dir, ParamData().mobility, False)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio)

        model =  RocketClassifier(
            num_kernels= ParamaRocketTrain().num_kernels,
            rocket_transform = "rocket",
            use_multivariate = "yes")

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
        with open(ParamDir().output_dir/data_name/(f"tsc_train_rocket_balanced.pickle"),"wb") as f:
            pickle.dump(results, f)

def kneighbors_trainer():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = Dataset(data_dir, ParamData().mobility, False)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio)

        model =  KNeighborsTimeSeriesClassifier(n_neighbors=8)
        param_grid = {"n_neighbors": [1, 5], "distance": ["euclidean", "dtw"]}
        parameter_tuning_method = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5))
        parameter_tuning_method.fit(X_train, y_train)

        y_pred = parameter_tuning_method.predict(X_test)

        results = {
            "estimator": parameter_tuning_method,
            "y_test": y_test,
            "y_pred": y_pred #np.array([y+1 for y in np.argmax(y_pred, axis=1)])
        }
        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/(f"tsc_train_kneighbors.pickle"),"wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    rocket_trainer()
    rocket_trainer_balanced()
    # kneighbors_trainer()