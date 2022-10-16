"""
implement ridge regression without discretization of coordinates
"""

from pathlib import Path

print(Path().absolute())

from modules.func import *
from modules.decoder import Results, RidgeRegression
from tqdm import tqdm
import pickle

# coordinate
coord_axis="y-axis"

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
    coords_xy,spikes=load_data(data_dir) # load coordinates and spikes data

    time_bin_size=1/3 #second
    n_time_bins,n_cells = spikes.shape

    design_mat_all=mk_design_matrix_decoder(spikes)
    coord=coords_xy[:,0] if coord_axis=="x-axis" else coords_xy[:,1]

    train_size=int(n_time_bins/2)

    X_train,y_train=design_mat_all[:train_size],coord[:train_size]
    X_test,y_test=design_mat_all[train_size:],coord[train_size:]

    results_list=[]
    # for p in range(10):
    for p in [2**i for i in range(3,21)]:
        rr=RidgeRegression()
        rr.fit(X_train,y_train,p)
        rr.predict(X_test)
        results=Results(rr)
        results_list.append(results.summary())

    # ---save theta(parameter) , prediction , test_data
    with open(output_dir/(f"rr_summary_coor{coord_axis}_{data_name}.pickle"),"wb") as f:
        pickle.dump([results_list,y_test],f)

