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
    coord_axis_opts=["x-axis","y-axis"]

    # data dir
    datalist = ParamDir().data_path_list

    # output dir
    output_dir = ParamDir().OUTPUT_DIR
    if not output_dir.exists():
        output_dir.mkdir()

    # get the regression results for all the mice
    for data_dir in tqdm(datalist[[0,2]]):
        data_name = str(data_dir).split('/')[-1]

        results_all=[]
        for nthist, coord_axis in product(ParamData().nthist_range, coord_axis_opts):
            design_matrix, coord = PastCoordDataset(data_dir, coord_axis, nthist).data # load coordinates and spikes data

            (X_train, y_train), (_, _) = spilt_data(design_matrix, coord, .8)

            search = SearchCV(RidgeRegression(), ParamTrain().scoring, ParamTrain().penalty_range, 10)
            search.evaluate_candidates(X_train, y_train)
            results_all.append((search.best_result, nthist, coord_axis))

        # ---save results
        with open(output_dir/(f"rr_past_coord_{data_name}.pickle"),"wb") as f:
            pickle.dump(results_all,f)

if __name__ == "__main__":
    main()