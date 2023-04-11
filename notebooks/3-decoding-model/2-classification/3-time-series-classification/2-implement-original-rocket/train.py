from typing import Tuple

from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import pickle
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.manifold import Isomap
from itertools import product

from dataloader.dataset import UniformSegmentDataset
from datasets import *
from param import *
from util import segment, downsample

def rocket_trainer_threshold_segment():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = UniformSegmentDataset(data_dir, ParamData().mobility, ParamData().shuffle)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio, ParamData().K)

        # rocket transform
        num_kernels = ParamData().num_kernels_KO if "KO" in data_name else ParamData().num_kernels_WT
        transform_pipeline = Pipeline([
            ("rocket", Rocket(num_kernels, random_state=ParamData().random_state)),
            ("std_scaler", StandardScaler()),
            ("l2_norm", Normalizer()),
        ])
        X_train = transform_pipeline.fit_transform(X_train)
        active_features = X_train.sum(axis=0)>0
        X_train = X_train[:, active_features]
        X_test = transform_pipeline.transform(X_test)
        X_test = X_test[:, active_features]

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

def rocket_trainer_tuning(data_dir, K_range, kernels_range, note):
    """The training script.
    """
    # for data_dir in tqdm(ParamDir().data_path_list):
    data_name = str(data_dir).split('/')[-1]
    res_all = []
    for K, num_kernels in product(K_range, kernels_range):
        dataset = UniformSegmentDataset(data_dir, ParamData().mobility, ParamData().shuffle)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio, K)

        # rocket transform
        transform_pipeline = Pipeline([
            ("rocket", Rocket(num_kernels, random_state=ParamData().random_state)),
            ("std_scaler", StandardScaler()),
            ("l2_norm", Normalizer()),
        ])
        X_train = transform_pipeline.fit_transform(X_train)
        active_features = X_train.sum(axis=0)>0
        X_train = X_train[:, active_features]
        X_test = transform_pipeline.transform(X_test)
        X_test = X_test[:, active_features]

        # normalization
        if X_train.shape[1]==0: 
            print(f"with K:{K} & num_kernels:{num_kernels}, found zero features")
            continue

        # cv tuning
        kfold = KFold(n_splits=ParamaRocketTrain().n_splits)
        model = RidgeClassifier(random_state=ParamData().random_state)
        clf = GridSearchCV(model, 
                            param_grid={"alpha": ParamaRocketTrain().alphas},
                            cv=kfold,
                            n_jobs=ParamaRocketTrain().njobs)
        clf.fit(X_train, y_train)

        # scoring
        scores = clf.score(X_test, y_test)

        res = {
            "estimator": clf, 
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
    # rocket_trainer_threshold_segment()
    # LEM_trainer_threshold_segment()
    # kneighbors_trainer()

    # ---- large scale tuning -----
    K_range = range(10, 22)
    kernels_range = [2**i for i in range(2, 13)]
    # # rocket_trainer_tuning(K_range, kernels_range, "large_scale")
    Parallel(n_jobs=ParamaRocketTrain().njobs)(delayed(
        rocket_trainer_tuning(data_dir, K_range, kernels_range, "large_scale")
        )(data_dir) for data_dir in tqdm(ParamDir().data_path_list))

    # ---- small scale tuning ----
    # K_range = [16]
    # kernels_range = range(100, 600, 20)
    # # rocket_trainer_tuning(K_range, kernels_range, "small_scale")
    # Parallel(n_jobs=-1)(delayed(
    #     rocket_trainer_tuning(data_dir, K_range, kernels_range, "small_scale")
    #     )(data_dir) for data_dir in tqdm(ParamDir().data_path_list))

    # ---- small scale tuning 2 ----
    # K_range = [20]
    # kernels_range = np.arange(1, 800, 5)
    # rocket_trainer_tuning(K_range, kernels_range, "small_scale_kernels")



    