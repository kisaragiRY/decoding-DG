from pathlib import Path
from modules.func import *
from modules.decoder import Results, RidgeRegression
from tqdm import tqdm
import pickle

#----variables
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
    binned_position=bin_pos(position,n_parts,partition_type)
    time_bin_size=1/3 #second
    n_time_bins,n_cells = spikes.shape

    design_mat_all=mk_design_matrix_decoder(spikes)

    train_size=int(n_time_bins/2)

    # # split train and test set for rolling origin cross validation
    # X_train_all,X_test_all=np.zeros((train_size,n_cells,n_time_bins-train_size)),np.zeros((1,n_cells,n_time_bins-train_size))
    # y_train_all,y_test_all=np.zeros((train_size,1,n_time_bins-train_size)),np.zeros((train_size,1,n_time_bins-train_size))

    # origin_range=range(n_time_bins-train_size)
    # for origin in origin_range:
    #     X_train_all[:,:,origin],y_train_all[:,:,origin]= design_mat_all[origin:origin+train_size] , binned_position[origin:origin+train_size].reshape(-1,1)
    #     X_test_all[:,:,origin], y_test_all[:,:,origin]= design_mat_all[origin+train_size+1] , binned_position[origin+train_size+1].reshape(-1,1)

    X_train,X_test=design_mat_all[:train_size],binned_position[:train_size]
    y_train,y_test=design_mat_all[train_size:],binned_position[train_size:]

    results_list=[]
    failed_penalty=[]
    # for p in range(10):
    for p in [2**i for i in range(3,13)]:
        rr=RidgeRegression()
        try: 
            theta=rr.fit(X_train, y_train,p)
            result=Results(rr)
            results_list.append(result.summary())
        except:
            print("fitting failed")
            failed_penalty.append(p)

    # ---save theta(parameter) , prediction , test_data
    with open(output_dir/(f"rr_predict_{data_name}_{partition_type}nParts{n_parts}.pickle"),"wb") as f:
        pickle.dump([results_list,y_test,failed_penalty],f)

