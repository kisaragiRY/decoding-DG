from tqdm import tqdm
import pickle

from train import Dataset
from param import *
# from decoder import SoftmaxRegression
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def main():
    """The training script.
    """
    for data_dir in tqdm(ParamDir().data_path_list):
        data_name = str(data_dir).split('/')[-1]
        
        dataset = Dataset(data_dir, False, False)

        (X_train, y_train), (X_test, y_test) = dataset.load_all_data(ParamData().window_size, ParamData().train_ratio)

        # with open(ParamDir().output_dir/data_name/(f"sm_training_firing_rate.pickle"),"rb") as f:
        #     (losses, beta) = pickle.load(f)

        model =  LogisticRegression(multi_class='multinomial', solver='newton-cg') #SoftmaxRegression() #sm.MNLogit(y_train.ravel(), X_train)

        # fit
        # result = model.fit(method="ncg")
        result = model.fit(X_train, y_train)

        # y_pred = model.predict(params=result.params ,exog = X_test) #model.predict(X_test, beta)
        y_pred = model.predict(X_test)
        print(np.unique(y_pred))

        results = {
            # "losses": losses,
            "estimator": model,
            "y_test": y_test,
            "y_pred": y_pred #np.array([y+1 for y in np.argmax(y_pred, axis=1)])
        }
        with open(ParamDir().output_dir/data_name/(f"sm_evaluation_firing_rate.pickle"),"wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main()