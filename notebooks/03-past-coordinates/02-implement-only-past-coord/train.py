from modules.func import *
from modules.decoder import RidgeRegression
from tqdm import tqdm
import pickle
from itertools import product
from typing import Tuple

from param import *
from modules.dataloader import PastCoordDataset
from modules.model_selection import SearchCV

def spilt_data(X: np.array, y: np.array, train_ratio: float) -> Tuple[Tuple[np.array,np.array],Tuple[np.array,np.array]]:
    """Get training and testing data."""
    train_size = int( len(X) * train_ratio )
    X_train,y_train = X[:train_size], y[:train_size]
    X_test,y_test = X[train_size:], y[train_size:]
    return (X_train, y_train), (X_test, y_test)


def main() -> None:
    # coordinate
    # coord_axis_opts=["x-axis",]
    coord_axis = "x-axis"

    # data dir
    datalist = ParamDir().data_path_list

    # output dir
    output_dir = ParamDir().output_dir

    # get the regression results for all the mice
    for data_dir in tqdm(datalist[[0,2]]):
        dataset = PastCoordDataset(data_dir)
        data_name = str(data_dir).split('/')[-1]

        results_all=[]
        for nthist in tqdm(ParamData().nthist_range):
            design_matrix, coord = dataset.load_all_data(coord_axis, nthist) # load coordinates and spikes data

            (X_train, y_train), (_, _) = spilt_data(design_matrix, coord, .8)

            search = SearchCV(ParamTrain().scoring, ParamTrain().penalty_range, ParamTrain().n_split)
            search.evaluate_candidates(X_train, y_train)
            results_all.append((search.best_result, nthist, coord_axis))

        # ---save results
        with open(output_dir/(f"rr_only_past_coord_{data_name}_{coord_axis}.pickle"),"wb") as f:
            pickle.dump(results_all,f)

if __name__ == "__main__":
    main()