from train import spilt_data
from param import *
from modules.dataloader import PastCoordDataset
from modules.decoder import RidgeRegression

import pickle
from tqdm import tqdm

def main():
    # output dir
    output_dir = ParamDir().output_dir
    datalist = ParamDir().data_path_list


    for data_dir in tqdm(datalist[[0,2]]):
        data_name = str(data_dir).split('/')[-1]
        dataset = PastCoordDataset(data_dir)
        with open(output_dir/(f"rr_only_past_coord_{data_name}.pickle"),"rb") as f:
            results_all = pickle.load(f)

        eval_results_all = []
        for result in results_all:
            design_matrix, coord = dataset.load_all_data(result["coord_axis"], result["nthist"]) # load coordinates and spikes data

            (_, _), (X_test, y_test) = spilt_data(design_matrix, coord, ParamTrain().train_size)

            rr = RidgeRegression()
            rr.load(result["fitted_param"])
            test_scores, y_pred = rr.evaluate(X_test, y_test, ParamEval().scoring)
            eval_result = {
                "eval_test_scores" : test_scores,
                "y_pred": y_pred,
                "y_test": y_test
            }
            eval_result.update(result)
            eval_results_all.append((eval_result))

            with open(output_dir/(f"rr_only_past_coord_eval_{data_name}.pickle"),"wb") as f:
                pickle.dump(eval_results_all,f)

if __name__ == "__main__":
    main()
