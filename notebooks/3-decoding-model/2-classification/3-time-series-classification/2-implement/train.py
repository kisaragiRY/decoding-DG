from typing import Tuple

from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import pickle
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score, KFold
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from itertools import product

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
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio, ParamData().K)

        # rocket transform
        rocket = Rocket(num_kernels=ParamData().num_kernels, 
                         random_state=ParamData().random_state)
        X_train = rocket.fit_transform(X_train).values
        active_features = X_train.sum(axis=0)>0
        X_train = X_train[:, active_features]
        X_test = rocket.transform(X_test).values
        X_test = X_test[:, active_features]

        # L2 normalization
        norm = Normalizer().fit(X_train)
        X_train = norm.transform(X_train)
        X_test = norm.transform(X_test)

        # cross validation
        kfold = KFold(n_splits=ParamaRocketTrain().n_splits)
        if ParamaRocketTrain().model_name == "Ridge":
            model = RidgeClassifier()
            clf = GridSearchCV(model, 
                            param_grid={"alpha": ParamaRocketTrain().alphas},
                            cv=kfold)
        elif ParamaRocketTrain().model_name == "SVM":
            model = SVC()
            clf = GridSearchCV(model, 
                            param_grid={"C": ParamaRocketTrain().Cs,
                                        "kernel": ["rbf", "sigmoid"]},
                            cv=kfold)
        elif ParamaRocketTrain().model_name == "Softmax":
            model = LogisticRegression(
                    multi_class='multinomial',
                    solver="newton-cg",
                    max_iter=1000,
                    n_jobs=-1)
            clf = GridSearchCV(model, 
                            param_grid={"C": ParamaRocketTrain().Cs},
                            cv=kfold)
        elif ParamaRocketTrain().model_name == "Kmeans":
            model = KMeans(
                    n_clusters=4,
                    random_state=ParamData().random_state)
            clf = GridSearchCV(model, 
                            param_grid={"algorithm": ["lloyd", "elkan"]},
                            cv=kfold)
        clf.fit(X_train, y_train)

        # scoring
        scores = clf.score(X_test, y_test)

        results = {
            "estimator": clf,
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

def rocket_trainer_tuning(data_dir, K_range, kernels_range, note):
    """The training script.
    """
    # for data_dir in tqdm(ParamDir().data_path_list):
    data_name = str(data_dir).split('/')[-1]
    res_all = []
    for K, num_kernels in product(K_range, kernels_range):
        dataset = ThresholdSegmentDataset(data_dir, ParamData().mobility, ParamData().shuffle)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio, K)

        # rocket transform
        rocket = Rocket(num_kernels, 
                        random_state=ParamData().random_state)
        X_train = rocket.fit_transform(X_train).values
        active_features = X_train.sum(axis=0)>0
        X_train = X_train[:, active_features]
        X_test = rocket.transform(X_test).values
        X_test = X_test[:, active_features]

        # normalization
        if X_train.shape[1]==0: 
            print(f"with K:{K} & num_kernels:{num_kernels}, found zero features")
            continue
        norm = Normalizer().fit(X_train)
        X_train = norm.transform(X_train)
        X_test = norm.transform(X_test)

        # cv tuning
        kfold = KFold(n_splits=ParamaRocketTrain().n_splits)
        model = RidgeClassifier()
        clf = GridSearchCV(model, 
                            param_grid={"alpha": ParamaRocketTrain().alphas},
                            cv=kfold)
        clf.fit(X_train, y_train)

        # scoring
        scores = clf.score(X_test, y_test)

        res = {
            "scores": scores,
            "K": K,
            "num_kernels": num_kernels,
        }
        res_all.append(res)
    if not (ParamDir().output_dir/data_name).exists():
        (ParamDir().output_dir/data_name).mkdir()
    with open(ParamDir().output_dir/data_name/(f"tsc_tuning_rocket_{note}.pickle"),"wb") as f:
        pickle.dump(res_all, f)

if __name__ == "__main__":
    # rocket_trainer()
    # rocket_trainer_balanced()
    rocket_trainer_threshold_segment()
    # LEM_trainer_threshold_segment()
    # kneighbors_trainer()

    # ---- large scale tuning -----
    # K_range = range(10, 81, 5)
    # kernels_range = [2**i for i in range(1, 11)]
    # # rocket_trainer_tuning(K_range, kernels_range, "large_scale")
    # Parallel(n_jobs=-1)(delayed(
    #     rocket_trainer_tuning(data_dir, K_range, kernels_range, "large_scale")
    #     )(data_dir) for data_dir in tqdm(ParamDir().data_path_list[2:]))

    # ---- small scale tuning ----
    # K_range = [20]
    # kernels_range = range(100, 500, 5)
    # rocket_trainer_tuning(K_range, kernels_range, "small_scale")

    # ---- small scale tuning 2 ----
    # K_range = [20]
    # kernels_range = np.arange(1, 800, 5)
    # rocket_trainer_tuning(K_range, kernels_range, "small_scale_kernels")



    