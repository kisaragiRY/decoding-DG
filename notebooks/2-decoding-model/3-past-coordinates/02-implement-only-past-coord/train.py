from modules.func import *
from modules.decoder import RidgeRegression
from tqdm import tqdm
import pickle
from itertools import product
from typing import Tuple

from param import *
from modules.dataloader import PastCoordDataset
from modules.metrics import get_scorer
from modules.model_selection import RidgeSigTest

def spilt_data(X: np.array, y: np.array, train_ratio: float) -> Tuple[Tuple[np.array,np.array],Tuple[np.array,np.array]]:
    """Get training and testing data."""
    train_size = int( len(X) * train_ratio )
    X_train,y_train = X[:train_size], y[:train_size]
    X_test,y_test = X[train_size:], y[train_size:]
    return (X_train, y_train), (X_test, y_test)


def main() -> None:
    # coordinate
    coord_axis_opts=["x-axis", "y-axis"]
    # coord_axis = "x-axis"

    # data dir
    datalist = ParamDir().data_path_list

    # output dir
    output_dir = ParamDir().output_dir

    # get the regression results for all the mice
    for data_dir in tqdm(datalist):
        dataset = PastCoordDataset(data_dir)
        data_name = str(data_dir).split('/')[-1]

        results_all=[]
        for nthist, coord_axis in tqdm(product(ParamData().nthist_range, coord_axis_opts)):
            design_matrix, coord = dataset.load_all_data(coord_axis, nthist) # load coordinates and spikes data

            (X_train, y_train), (X_test, y_test) = spilt_data(design_matrix, coord, .8)

            scorer = get_scorer(ParamTrain().scoring)
            # train with penalty 0 (no need for cross validation)
            rr = RidgeRegression()
            rr.fit(X_train, y_train, 0)
            rr.predict(X_test)
            result = {
                "train_scores": scorer(y_train, np.einsum("ij,j->i",X_train, rr.fitted_param)),
                "test_scores" : scorer(y_test, rr.prediction),
                "fitted_param": rr.fitted_param,
                "hyper_param": 0,
                }

            # significance results
            sig_tests = RidgeSigTest(rr)
            more_results ={
                "RSS": sig_tests.RSS,
                "F_stat": sig_tests.f_stat,
                "F_p_value": sig_tests.f_p_value,
                "coeff_stats": sig_tests.t_stat_list,
                "coeff_p_values": sig_tests.t_p_value_list,
                "nthist": nthist, 
                "coord_axis": coord_axis
            }
            result.update(more_results)
            results_all.append(result)

        # ---save results
        with open(output_dir/(f"rr_only_past_coord_{data_name}.pickle"),"wb") as f:
            pickle.dump(results_all,f)

if __name__ == "__main__":
    main()