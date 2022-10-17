"""
implement ridge regression using summed spikes, 
see mk_design_matrix_decoder2 in func.py for details.
"""

from pathlib import Path
from modules.func import *
from modules.decoder import Results, RidgeRegression
from tqdm import tqdm
import pickle

# coordinate
coord_axis="y-axis"

# range of nthist(number of time bins for history)
nthist_range=[3*i for i in range(1,17)[::2]]

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

    results_nthist=[]
    for nthist in nthist_range:
        design_mat_all=mk_design_matrix_decoder2(spikes,nthist) # design matrix with summed spikes
        coord=coords_xy[:,0][nthist:] if coord_axis=="x-axis" else coords_xy[:,1][nthist:] # the length would shrink because of the nthist

        train_size=int(n_time_bins/2)

        X_train,y_train=design_mat_all[:train_size],coord[:train_size]
        X_test,y_test=design_mat_all[train_size:],coord[train_size:]

        results_list=[]
        penalty_range=[2**i for i in range(3,21)]
        for p in penalty_range:
            rr=RidgeRegression()
            rr.fit(X_train,y_train,p)
            rr.predict(X_test)
            results=Results(rr)
            results_list.append(results.summary())
        
        results_nthist.append((results_list,nthist))

    # ---save results
    with open(output_dir/(f"rr_summed_spikes_coor{coord_axis}_{data_name}.pickle"),"wb") as f:
        pickle.dump(results_nthist,f)

