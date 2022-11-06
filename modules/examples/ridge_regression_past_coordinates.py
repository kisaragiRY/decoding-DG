from pathlib import Path
from modules.func import *
from modules import Results, RidgeRegression
from tqdm import tqdm
import pickle
from itertools import product


# coordinate
coord_axis_opts=["x-axis","y-axis"]

# range of nthist(number of time bins for history)
nthist_range=[3*i for i in range(17)[::2]]

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

    results_all=[]
    for nthist,coord_axis in product(nthist_range,coord_axis_opts):
        coord=coords_xy[:,0][nthist:] if coord_axis=="x-axis" else coords_xy[:,1][nthist:] # the length would shrink because of the nthist
        design_mat_all=mk_design_matrix_decoder3(spikes, coord, nthist) # design matrix with past coordinates

        train_size=int(n_time_bins/2)
        X_train,y_train=design_mat_all[:train_size],coord[:train_size]
        X_test,y_test=design_mat_all[train_size:],coord[train_size:]

        penalty_range=[i for i in np.arange(4,16,.2)]
        for p in penalty_range:
            rr=RidgeRegression()
            rr.fit(X_train,y_train,p)
            rr.predict(X_test)
            model_smry=Results(rr).summary()

            result_wrap={
                "model_smry": model_smry,
                "nthist": nthist,
                "penalty": p,
                "coord_axis": coord_axis,
                "y_test": y_test
            }
            results_all.append(result_wrap)

    # ---save results
    with open(output_dir/(f"rr_past_coord_{data_name}.pickle"),"wb") as f:
        pickle.dump(results_all,f)

