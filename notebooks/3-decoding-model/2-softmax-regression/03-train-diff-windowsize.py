from tqdm import tqdm
import pickle

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

from train import Dataset
from param import *

def main():
    """The training script.

    Train with downsampling.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]
        
        dataset = Dataset(data_dir, False, False)

        results_all = []
        for window_size in ParamData().window_size:

            (X_train, y_train), (X_test, y_test) = dataset.load_all_data(window_size, ParamData().train_ratio)

            model =  LogisticRegression(multi_class='multinomial', solver='newton-cg') #SoftmaxRegression() #sm.MNLogit(y_train.ravel(), X_train)

            # fit
            # result = model.fit(method="ncg")
            result = model.fit(X_train, y_train)

            # y_pred = model.predict(params=result.params ,exog = X_test) #model.predict(X_test, beta)
            y_pred = model.predict(X_test)

            results = {
                "window_size": window_size,
                "estimator": model,
                "y_test": y_test,
                "y_pred": y_pred #np.array([y+1 for y in np.argmax(y_pred, axis=1)])
            }
            results_all.append(results)
        with open(ParamDir().output_dir/data_name/(f"sm_train_diff_windowsize.pickle"),"wb") as f:
            pickle.dump(results_all, f)

if __name__ == "__main__":
    main()