from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sktime.transformations.panel.rocket import Rocket
from itertools import product
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm

from param import *
from dataloader.dataset import UniformSegmentDataset



def rocket_trainer_tuning(data_dir, K_range, kernels_range, note):
    """The training script.
    """
    data_name = str(data_dir).split('/')[-1]
    res_all = []
    for K, num_kernels in tqdm(product(K_range, kernels_range)):
        dataset = UniformSegmentDataset(data_dir, ParamData().mobility, ParamData().shuffle, ParamData().random_state)
        (X_train, y_train) = dataset.load_all_data(ParamData().window_size, K)

        # transform
        transform_pipeline = Pipeline([
            ("rocket", Rocket(num_kernels, random_state=ParamData().random_state)),
            ("std_scaler", StandardScaler()),
            ("l2_norm", Normalizer()),
        ])
        X_train = transform_pipeline.fit_transform(X_train)
        X_train = X_train[:, X_train.sum(axis=0)>0]

        
        if X_train.shape[1]==0: 
            print(f"with K:{K} & num_kernels:{num_kernels}, found zero features")
            continue

        # K means training
        model = KMeans(n_clusters=ParamTrain().n_splits, 
                       random_state=ParamTrain().random_state)
        model.fit(X_train, y_train)


        res = {
            "estimator": model, 
            "X_shape": X_train.shape,
            "K": K,
            "num_kernels": num_kernels,
        }
        res_all.append(res)
    if not (ParamDir().output_dir/data_name).exists():
        (ParamDir().output_dir/data_name).mkdir()
    with open(ParamDir().output_dir/data_name/(f"tsc_tuning_k_means_{note}.pickle"),"wb") as f:
        pickle.dump(res_all, f)

if __name__ == "__main__":
    # ---- large scale tuning -----
    K_range = range(10, 22)
    kernels_range = [2**i for i in range(2, 11)]
    # # rocket_trainer_tuning(K_range, kernels_range, "large_scale")
    Parallel(n_jobs=-1)(delayed(
        rocket_trainer_tuning(data_dir, K_range, kernels_range, "large_scale")
        )(data_dir) for data_dir in ParamDir().data_path_list)

    # ---- small scale tuning ----
    # K_range = [16]
    # kernels_range = range(100, 600, 20)
    # # rocket_trainer_tuning(K_range, kernels_range, "small_scale")
    # Parallel(n_jobs=-1)(delayed(
    #     rocket_trainer_tuning(data_dir, K_range, kernels_range, "small_scale")
    #     )(data_dir) for data_dir in tqdm(ParamDir().data_path_list))