from typing import Tuple

from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import pickle
from sktime.transformations.panel.rocket import Rocket
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from itertools import product
from numba import jit, prange

from dataloader.dataset import UniformSegmentDataset
from datasets import *
from param import *

def rocket_trainer_threshold_segment():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = UniformSegmentDataset(data_dir, ParamData().mobility, ParamData().shuffle, ParamData().random_state)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().K, ParamData().train_ratio)

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
        print(X_train.shape)

        # cross validation
        kfold = KFold(n_splits=ParamaRocketTrain().n_splits)
        if ParamaRocketTrain().model_name == "Ridge":
            model = RidgeClassifier(random_state=ParamaRocketTrain().random_state)
            clf = GridSearchCV(model, 
                            param_grid={"alpha": ParamaRocketTrain().alphas},
                            cv=kfold,
                            scoring="f1_micro")
        elif ParamaRocketTrain().model_name == "SVM":
            model = SVC(random_state=ParamaRocketTrain().random_state)
            clf = GridSearchCV(model, 
                            param_grid={"C": ParamaRocketTrain().Cs,
                                        "kernel": ["rbf", "sigmoid"]},
                            cv=kfold,
                            scoring="f1_micro")
        clf.fit(X_train, y_train)

        # scoring
        scores = clf.score(X_test, y_test)

        results = {
            "estimator": clf,
            "scores": scores,
            "data": [(X_train, y_train), (X_test, y_test)]
        }
        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/
                  (f"tsc_f1_train_rocket_{ParamaRocketTrain().model_name}_threshold_segment_{ParamData().shuffle}.pickle"),
                  "wb") as f:
            pickle.dump(results, f)

def rocket_shuffle_trainer(data_dir: Path, repeats: int) -> None:
    data_name = str(data_dir).split('/')[-1]
    res_all = []
    for seed in tqdm(prange(repeats)):
        dataset = UniformSegmentDataset(data_dir, ParamData().mobility, "segment label shuffling", seed)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().K, ParamData().train_ratio)

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
            model = RidgeClassifier(random_state=ParamaRocketTrain().random_state)
            clf = GridSearchCV(model, 
                            param_grid={"alpha": ParamaRocketTrain().alphas},
                            cv=kfold,
                            scoring="f1_micro")
        elif ParamaRocketTrain().model_name == "SVM":
            model = SVC(random_state=ParamaRocketTrain().random_state)
            clf = GridSearchCV(model, 
                            param_grid={"C": ParamaRocketTrain().Cs,
                                        "kernel": ["rbf", "sigmoid"]},
                            cv=kfold,
                            scoring="f1_micro")
        clf.fit(X_train, y_train)

        # scoring
        scores = clf.score(X_test, y_test)

        res = {
            "estimator": clf,
            "scores": scores,
            "seed": seed,
            "data": [(X_train, y_train), (X_test, y_test)]
        }
        res_all.append(res)
    if not (ParamDir().output_dir/data_name).exists():
        (ParamDir().output_dir/data_name).mkdir()
    with open(ParamDir().output_dir/data_name/(f"tsc_shuffle_f1_{ParamaRocketTrain().model_name}_{ParamData().shuffle}_train_rocket.pickle"),"wb") as f:
        pickle.dump(res_all, f)


if __name__ == "__main__":
    rocket_trainer_threshold_segment()

    # ---- shuffle train ----
    repeats = 1000
    Parallel(n_jobs=15)(delayed(
        rocket_shuffle_trainer(data_dir, repeats)
        )(data_dir) for data_dir in tqdm(ParamDir().data_path_list))





    