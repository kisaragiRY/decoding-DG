from dataloader import SmoothedSpikesDataset
from decoder import RidgeRegression

from param import *
import pickle
from tqdm import tqdm

def main():

    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]
        
        with open(ParamDir().output_dir/data_name/(f"rr_firing_rate.pickle"),"rb") as f:
            results_all = pickle.load(f)

        eval_results_all = []
        for result in tqdm(results_all):
            dataset = SmoothedSpikesDataset(data_dir, result["coord_axis"], False, False)
            (_, _), (X_test, y_test) = dataset.load_all_data(
                result["window_size"], 
                ParamTrain().train_size)  # load coordinates and spikes data

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
