from train import spilt_data
from param import *
from dataloader import SmoothedSpikesDataset
from decoder import RidgeRegression

import pickle
from tqdm import tqdm

def main():

    for data_dir in tqdm(ParamDir().data_path_list[1:]):
        data_name = str(data_dir).split('/')[-1]
        dataset = SmoothedSpikesDataset(data_dir)
        with open(ParamDir().output_dir/(f"rr_smoothed_spikes_{data_name}.pickle"),"rb") as f:
            results_all = pickle.load(f)

        eval_results_all = []
        for result in tqdm(results_all):
            design_matrix, coord = dataset.load_all_data(result["coord_axis"], result["nthist"], result["window_size"])  # load coordinates and spikes data

            (_, _), (X_test, y_test) = spilt_data(design_matrix, coord, ParamTrain().train_size)

            rr = RidgeRegression()
            rr.load(result["estimator"].fitted_param)
            test_scores, y_pred = rr.evaluate(X_test, y_test, ParamEval().scoring)
            eval_results = {
                "eval_test_scores" : test_scores,
                "y_pred": y_pred,
                "y_test": y_test
            }
            eval_results.update(result)
            eval_results_all.append(eval_results)

            with open(ParamDir().output_dir/(f"rr_smoothed_spikes_eval_{data_name}.pickle"),"wb") as f:
                pickle.dump(eval_results_all,f)

if __name__ == "__main__":
    main()
