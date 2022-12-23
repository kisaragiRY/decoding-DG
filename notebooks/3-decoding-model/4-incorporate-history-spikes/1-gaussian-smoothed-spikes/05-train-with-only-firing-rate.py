from func import *
from tqdm import tqdm
import pickle
from itertools import product
from typing import Tuple

from param import *
from dataloader import SmoothedSpikesDataset
from model_selection import SearchCV
from util import spilt_data


def main() -> None:

    # get the regression results for all the mice
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]
        dataset = SmoothedSpikesDataset(data_dir, False)

        results_all=[]
        for coord_axis, window_size, sigma in product( ParamFiringRateData().coord_axis_opts, ParamFiringRateData().window_size_range, ParamFiringRateData().sigma_range):
            design_matrix, coord = dataset.load_all_data(
                coord_axis, 
                ParamFiringRateData().nthist, 
                window_size,
                sigma
                ) # load coordinates and firing rate data

            (X_train, y_train), (_, _) = spilt_data(design_matrix, coord, ParamTrain().train_size)

            search = SearchCV(ParamTrain().scoring, ParamTrain().penalty_range, ParamTrain().n_split)
            search.evaluate_candidates(X_train, y_train)

            train_results = {
                "sigma": sigma,
                "coord_axis": coord_axis,
                "window_size": window_size
            }
            train_results.update(search.best_result)
            results_all.append((train_results))

        # ---save results
        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/(f"rr_firing_rate.pickle"),"wb") as f:
            pickle.dump(results_all,f)

if __name__ == "__main__":
    main()