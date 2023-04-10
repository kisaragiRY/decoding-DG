import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from itertools import product
from joblib import Parallel, delayed

from dataloader.dataset import UniformSegmentDataset
from param import *
from rocket import Rocket

def modified_rocket_trainer():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]

        dataset = UniformSegmentDataset(data_dir, ParamData().mobility, ParamData().shuffle)
        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio, ParamData().K)

        # rocket transform
        num_kernels = ParamData().num_kernels_KO if "KO" in data_name else ParamData().num_kernels_WT
        transform_pipeline = Pipeline([
            ("rocket", Rocket(num_kernels, kernel_dim= ParamData().kernel_dim, random_state=ParamData().random_state, njobs=ParamData().njobs)),
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
                  (f"tsc_train_modified_rocket_{ParamaRocketTrain().model_name}_threshold_segment_{ParamData().shuffle}.pickle"),
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
            ("rocket", Rocket(num_kernels, kernel_dim= ParamData().kernel_dim, random_state=ParamData().random_state, njobs=ParamData().njobs)),
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
                            cv=kfold)
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
    with open(ParamDir().output_dir/data_name/(f"tsc_tuning_modified_rocket_{note}.pickle"),"wb") as f:
        pickle.dump(res_all, f)


if __name__ == "__main__":
    modified_rocket_trainer()

    # ---- large scale tuning -----
    # K_range = [16]
    # kernels_range = [2**i for i in range(2, 13)]
    # # # rocket_trainer_tuning(K_range, kernels_range, "large_scale")
    # Parallel(n_jobs=-1)(delayed(
    #     rocket_trainer_tuning(data_dir, K_range, kernels_range, "large_scale")
    #     )(data_dir) for data_dir in tqdm(ParamDir().data_path_list))

    # ---- small scale tuning ----
    # K_range = [16]
    # kernels_range = range(900, 1500, 20)
    # # rocket_trainer_tuning(K_range, kernels_range, "small_scale")
    # Parallel(n_jobs=-1)(delayed(
    #     rocket_trainer_tuning(data_dir, K_range, kernels_range, "small_scale")
    #     )(data_dir) for data_dir in tqdm(ParamDir().data_path_list))



    