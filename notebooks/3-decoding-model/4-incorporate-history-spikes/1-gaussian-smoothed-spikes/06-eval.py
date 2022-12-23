from dataloader import SmoothedSpikesDataset
from decoder import RidgeRegression

from util import spilt_data
from param import *
import pickle
from tqdm import tqdm

def main():

    for data_dir in tqdm(ParamDir().data_path_list[[0]]):
        data_name = str(data_dir).split('/')[-1]
        dataset = SmoothedSpikesDataset(data_dir, False)
        with open(ParamDir().output_dir/data_name/(f"rr_firing_rate.pickle"),"rb") as f:
            results_all = pickle.load(f)

        eval_results_all = []
        for result in tqdm(results_all):
            design_matrix, coord = dataset.load_all_data(
                result["coord_axis"], 
                ParamFiringRateData().nthist, 
                result["window_size"], 
                result["sigma"])  # load coordinates and spikes data

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

            with open(ParamDir().output_dir/data_name/(f"rr_firing_rate_eval.pickle"),"wb") as f:
                pickle.dump(eval_results_all,f)

if __name__ == "__main__":
    main()
