from typing import Tuple

from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# from modules.dataloader.dataset import UniformSegmentDataset
from datasets import *
from param import *

def logistic_regression_trainer():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = ConcatDataset(data_dir, ParamData().mobility, ParamData().shuffle, ParamData().random_state)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().K, ParamData().train_ratio)

        # normalization & standalization
        transform_pipeline = Pipeline([
            # ("std_scaler", StandardScaler()),
            ("l2_norm", Normalizer()),
        ])
        X_train = transform_pipeline.fit_transform(X_train)
        active_features = X_train.sum(axis=0)>0
        X_train = X_train[:, active_features]
        X_test = transform_pipeline.transform(X_test)
        X_test = X_test[:, active_features]
        print(X_train.shape)

        # cross validation
        kfold = KFold(n_splits=ParamaLinearTrain().n_splits)
        model = LogisticRegression(
                multi_class='multinomial',
                solver="newton-cg",
                max_iter=1000,
                n_jobs=-1,
                random_state=ParamaLinearTrain().random_state)
        clf = GridSearchCV(model, 
                            param_grid={"C": np.logspace(-3, 3, 10)},
                            cv=kfold)
        clf.fit(X_train, y_train)
        scores = clf.score(X_test, y_test)
        # model2 = LinearSVC(random_state=ParamaLinearTrain().random_state)                   


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
                  (f"tsc_train_linear_logistic_{ParamaLinearTrain().model_name}_{ParamData().shuffle}.pickle"),
                  "wb") as f:
            pickle.dump(results, f)

def logistic_shuffle_trainer(data_dir: Path, repeats: int):
    data_name = str(data_dir).split('/')[-1]
    res_all = []
    for seed in tqdm(range(repeats)):
        dataset = ConcatDataset(data_dir, ParamData().mobility, "segment label shuffling", seed)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().K, ParamData().train_ratio)

        transform_pipeline = Pipeline([
            ("l2_norm", Normalizer()),
        ])
        X_train = transform_pipeline.fit_transform(X_train)
        active_features = X_train.sum(axis=0)>0
        X_train = X_train[:, active_features]
        X_test = transform_pipeline.transform(X_test)
        X_test = X_test[:, active_features]

        # cross validation
        kfold = KFold(n_splits=ParamaLinearTrain().n_splits)
        model = LogisticRegression(
                multi_class='multinomial',
                solver="newton-cg",
                max_iter=1000,
                n_jobs=-1,
                random_state=ParamaLinearTrain().random_state)
        clf = GridSearchCV(model, 
                            param_grid={"C": np.logspace(-3, 3, 10)},
                            cv=kfold)
        clf.fit(X_train, y_train)
        scores = clf.score(X_test, y_test)

        res = {
            "estimator": clf,
            "scores": scores,
            "seed": seed,
        }
        res_all.append(res)
    if not (ParamDir().output_dir/data_name).exists():
        (ParamDir().output_dir/data_name).mkdir()
    with open(ParamDir().output_dir/data_name/(f"tsc_shuffle_logistic_{ParamData().shuffle}_train_rocket.pickle"),"wb") as f:
        pickle.dump(res_all, f)

def main():
    # logistic_regression_trainer()
    repeats = 1000
    Parallel(n_jobs=15)(delayed(
        logistic_shuffle_trainer(data_dir, repeats)
        )(data_dir) for data_dir in tqdm(ParamDir().data_list))

if __name__ == "__main__":
    main()