from func import *
from tqdm import tqdm
import pickle
from itertools import product
from typing import Tuple

from param import *
from dataloader import SmoothedSpikesDataset
from model_selection import SearchCV
from utils.util import spilt_data


def main() -> None:

    # get the regression results for all the mice
    for data_dir in tqdm(ParamDir().data_path_list[1:]):
        data_name = str(data_dir).split('/')[-1]
        dataset = SmoothedSpikesDataset(data_dir)

        results_all=[]
        for nthist, coord_axis, window_size in product(ParamPastCoordData().nthist_range, ParamPastCoordData().coord_axis_opts, ParamPastCoordData().window_size_range):
            design_matrix, coord = dataset.load_all_data(coord_axis, nthist, window_size) # load coordinates and spikes data

            (X_train, y_train), (_, _) = spilt_data(design_matrix, coord, .8)

            search = SearchCV(ParamTrain().scoring, ParamTrain().penalty_range, ParamTrain().n_split)
            search.evaluate_candidates(X_train, y_train)

            train_results = {
                "nthist": nthist, 
                "coord_axis": coord_axis,
                "window_size": window_size
            }
            train_results.update(search.best_result)
            results_all.append((train_results))

        # ---save results
        with open(ParamDir().output_dir/(f"rr_smoothed_spikes_{data_name}.pickle"),"wb") as f:
            pickle.dump(results_all,f)

if __name__ == "__main__":
    main()