from pathlib import Path
from modules.func import *
from modules.decoder import Results, RidgeRegression
from tqdm import tqdm
import pickle
from itertools import product
from typing import Tuple
from sklearn.model_selection import TimeSeriesSplit

from param import *
from modules.dataloader import PastCoordDataset
from modules.model_selection import SearchCV

def spilt_data(X: np.array, y: np.array, train_ratio: float) -> Tuple[Tuple[np.array,np.array],Tuple[np.array,np.array]]:
    """Get training and testing data."""
    train_size = int( len(X) * train_ratio )
    X_train,y_train = X[:train_size], y[:train_size]
    X_test,y_test = X[train_size:], y[train_size:]
    return (X_train, y_train), (X_test, y_test)

def get_val_data(train_data: Tuple[np.array, np.array], n_fold: int) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """Get validation data from X.
    
    Based on rolling origin cross validation techniques,
    one fold from n-fold of data would be used for validating,
    and the rest would be used to training.

    Parameter
    ---------
    train_data: Tuple[np.array, np.array]
        the training data that includes X_train and y_train
    n_fold: int
        the number of folds to split the data
    
    Return
    ---------
    X_train: np.array
        training data
    X_val: np.array
        validation data
    """
    X, y = train_data
    fold_size = int( len(X) / n_fold )
    for id_fold in range(n_fold-1):
       id_index = ( id_fold + 1 ) * fold_size
       X_train, X_val= X[:id_index], X[id_index:id_index+fold_size]
       y_train, y_val= y[:id_index], y[id_index:id_index+fold_size]
       yield (X_train, X_val), (y_train, y_val)

def main() -> None:
    # coordinate
    coord_axis_opts=["x-axis","y-axis"]

    # data dir
    datalist = ParamDir().datalist

    # output dir
    output_dir = ParamDir().OUTPUT_DIR
    if not output_dir.exists():
        output_dir.mkdir()

    # get the regression results for all the mice
    for data_dir in tqdm(datalist):
        data_name = str(data_dir).split('/')[-1]

        results_all=[]
        for nthist, coord_axis in product(ParamData().nthist_range, coord_axis_opts):
            design_matrix, coord = PastCoordDataset(data_dir, coord_axis, nthist).data # load coordinates and spikes data

            X, y, (X_test, y_test) = spilt_data(design_matrix, coord, .8)

            # rolling origin cross validation(named time series split in sklearn)
            tscv = TimeSeriesSplit(n_splits=10)

            search = SearchCV(RidgeRegression(), ParamTrain().scoring, ParamTrain().penalty_range, tscv)
            
            for id_, (train_index, test_index) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                p = id_ * .2 + 4 # penalty  

                rr = RidgeRegression()
                rr.fit(X_train,y_train,p)
                model_smry = Results(rr).summary()

                result_wrap = {
                    "model_smry": model_smry,
                    "nthist": nthist,
                    "penalty": p,
                    "coord_axis": coord_axis,
                }
                results_all.append(result_wrap)

        # ---save results
        with open(output_dir/(f"rr_past_coord_{data_name}.pickle"),"wb") as f:
            pickle.dump(results_all,f)

if __name__ == "__main__":
    main()