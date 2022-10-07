from pathlib import Path
from modules.func import *
from modules.decoder import RidgeRegression
from tqdm import tqdm
import pickle

#----variables
decoder_m="ridge regression" # decoder method
partition_type="vertical"
n_parts=4

# data dir
all_data_dir=Path('data/alldata/')
datalist=[x for x in all_data_dir.iterdir()]

# output dir
output_dir=Path("output/data/ridge_regression/")
if not output_dir.exists():
    output_dir.mkdir()

# get the regression results for all the mice
for data_dir in tqdm(datalist):
    data_name=str(data_dir).split('/')[-1]
    position,spikes=load_data(data_dir) # load data

    # binned_position=bin_pos(position,n_parts,partition_type)
    binned_position=position
    time_bin_size=1/3 #second
    n_time_bins,n_cells = spikes.shape

    design_mat_all=mk_design_matrix_decoder(spikes)

    # split train and test set for cross validation
    train_size=int(n_time_bins/2)
    X_train_all,X_test_all=np.zeros((train_size,n_cells,n_time_bins-train_size)),np.zeros((1,n_cells,n_time_bins-train_size))
    y_train_all,y_test_all=np.zeros((train_size,1,n_time_bins-train_size)),np.zeros((train_size,1,n_time_bins-train_size))

    
    for i in train_size:
        X_train_all[:,:,i]= design_mat_all[:n_time_bins_train] , binned_position[:n_time_bins_train].reshape(-1,1)
        design_mat_test, binned_position_test = design_mat_all[n_time_bins_train:] , binned_position[n_time_bins_train:].reshape(-1,1)


    theta_prediction_penalty=[]
    failed_penalty=[]
    # for p in range(10):
    for p in [2**i for i in range(3,13)]:
        rr=RidgeRegression()
        try: 
            theta=rr.fit(design_mat_train, binned_position_train,p)
            prediction=rr.predict(design_mat_test)
            prediction_train=rr.predict(design_mat_train)
        except:
            print("fitting failed")
            failed_penalty.append(p)
            # if fitting failed, set the following variables to np.nan
            theta=np.array([np.nan]*design_mat_train.shape[1])
            prediction=np.array([np.nan]*len(binned_position_test))
            prediction_train=np.array([np.nan]*len(binned_position_test))
        theta_prediction_penalty.append([theta,prediction,prediction_train,p])

    # ---save theta(parameter) , prediction , test_data
    # with open(output_dir/(f"lgr_predict_{data_name}.pickle"),"wb") as f:
    # with open(output_dir/(f"lgr_predict_{data_name}_withLargerPenalty_{n_parts}_{partition_type}.pickle"),"wb") as f:
    with open(output_dir/(f"lgr_predict_{data_name}_withoutPartition.pickle"),"wb") as f:
        pickle.dump([theta_prediction_penalty,binned_position_test,binned_position_train,failed_penalty],f)



