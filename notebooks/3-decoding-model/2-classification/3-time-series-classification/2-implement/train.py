from typing import Tuple

import pandas as pd
from tqdm import tqdm
import pickle
from sktime.transformations.panel.rocket import Rocket
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score, KFold
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC

from datasets import *
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

        dataset = BalancedSegmentDataset(data_dir, ParamData().mobility, False)
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

def rocket_trainer_threshold_segment():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = ThresholdSegmentDataset(data_dir, ParamData().mobility, ParamData().shuffle)
        X_train, y_train = dataset.load_all_data(ParamData().window_size, ParamData().K)

        # rocket transform
        X_train = Rocket(num_kernels=ParamData().num_kernels, 
                         random_state=ParamData().random_state).fit_transform(X_train)

        # model =  RocketClassifier(
        #     num_kernels= ParamaRocketTrain().num_kernels,
        #     rocket_transform = "rocket",
        #     use_multivariate = "yes")
        if ParamaRocketTrain().model_name == "Ridge":
            model = RidgeClassifier()
        elif ParamaRocketTrain().model_name == "SVM":
            model = SVC()
        elif ParamaRocketTrain().model_name == "Softmax":
            model = LogisticRegression(
                    multi_class='multinomial',
                    solver="newton-cg",
                    max_iter=1000,
                    n_jobs=-1)

        # cross validation
        kfold = KFold(n_splits=ParamaRocketTrain().n_splits)
        scores = cross_val_score(model, X_train, y_train, cv=kfold)

        results = {
            "estimator": model,
            "scores": scores
        }
        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/
                  (f"tsc_train_rocket_{ParamaRocketTrain().model_name}_threshold_segment_{ParamData().shuffle}.pickle"),
                  "wb") as f:
            pickle.dump(results, f)

def LEM_trainer_threshold_segment():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = DimRedDataset(data_dir, ParamData().mobility, ParamData().shuffle)
        X_train, y_train = dataset.load_all_data(ParamData().window_size, ParamData().reduction_method)

        if ParamaRocketTrain().model_name == "Ridge":
            model = RidgeClassifier()
        elif ParamaRocketTrain().model_name == "SVM":
            model = SVC()
        elif ParamaRocketTrain().model_name == "Softmax":
            model = LogisticRegression(
                    multi_class='multinomial',
                    solver="newton-cg",
                    max_iter=1000,
                    n_jobs=-1)

        # cross validation
        kfold = KFold(n_splits=ParamaRocketTrain().n_splits)
        scores = cross_val_score(model, X_train, y_train, cv=kfold)

        results = {
            "estimator": model,
            "scores": scores
        }
        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/
                  (f"tsc_train_LEM_{ParamaRocketTrain().model_name}_threshold_segment_{ParamData().shuffle}.pickle"),
                  "wb") as f:
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
    # rocket_trainer()
    # rocket_trainer_balanced()
    # rocket_trainer_threshold_segment()
    LEM_trainer_threshold_segment()
    # kneighbors_trainer()