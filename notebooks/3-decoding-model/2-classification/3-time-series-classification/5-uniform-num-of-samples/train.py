from typing import Tuple

from joblib import Parallel, delayed
from dataclasses import asdict
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

from dataset import ResampledDataset
from param import *

def rocket_trainer():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = ResampledDataset(data_dir, ParamData().mobility, ParamData().shuffle, ParamData().random_state, ParamData().stand_y_classes, ParamData().num_samples)
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
                            cv=kfold)
        elif ParamaRocketTrain().model_name == "SVM":
            model = SVC(random_state=ParamaRocketTrain().random_state)
            clf = GridSearchCV(model, 
                            param_grid={"C": ParamaRocketTrain().Cs,
                                        "kernel": ["rbf", "sigmoid"]},
                            cv=kfold)
        elif ParamaRocketTrain().model_name == "Softmax":
            model = LogisticRegression(
                    multi_class='multinomial',
                    solver="newton-cg",
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=ParamaRocketTrain().random_state)
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
                  (f"tsc_train_rocket_{ParamaRocketTrain().model_name}_uniform_num_of_samples_{ParamData().shuffle}.pickle"),
                  "wb") as f:
            pickle.dump(results, f)

def rocket_shuffle_trainer(data_dir: Path, repeats: int) -> None:
    data_name = str(data_dir).split('/')[-1]
    res_all = []
    for seed in tqdm(prange(repeats)):
        dataset = ResampledDataset(data_dir, ParamData().mobility, ParamData().shuffle, seed, ParamData().stand_y_classes, ParamData().num_samples)
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
                            cv=kfold)
        elif ParamaRocketTrain().model_name == "SVM":
            model = SVC(random_state=ParamaRocketTrain().random_state)
            clf = GridSearchCV(model, 
                            param_grid={"C": ParamaRocketTrain().Cs,
                                        "kernel": ["rbf", "sigmoid"]},
                            cv=kfold)
        clf.fit(X_train, y_train)

        # scoring
        scores = clf.score(X_test, y_test)

        res = {
            "estimator": clf,
            "scores": scores,
            "seed": seed,
        }
        res_all.append(res)
    if not (ParamDir().output_dir/data_name).exists():
        (ParamDir().output_dir/data_name).mkdir()
    with open(ParamDir().output_dir/data_name/(f"tsc_shuffle_uniform_num_of_samples_{ParamaRocketTrain().model_name}_{ParamData().shuffle}_train_rocket.pickle"),"wb") as f:
        pickle.dump(res_all, f)


if __name__ == "__main__":
    # rocket_trainer()

    # ---- shuffle train ----
    repeats = 1000
    Parallel(n_jobs=15)(delayed(
        rocket_shuffle_trainer(data_dir, repeats)
        )(data_dir) for data_dir in tqdm(ParamDir().data_list))





    