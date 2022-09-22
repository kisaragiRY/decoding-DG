import numpy as np
from scipy.linalg import inv
from pathlib import Path
import pickle
from func import *
from tqdm import tqdm

class linear_gaussian():
    '''
    a linear guassian model
    x_t=theta.T・n_t + b_t
    x_t: discretized position
    theta: parameter
    n_t: spikes
    b_t: intercept
    '''
    def __init__(self) -> None:
        pass

    def fit(self,design_matrix_train:np.array,binned_position_train:np.array):
        '''
        fitting based on training data
        '''
        tmp1=np.einsum("ji,ik->jk",design_matrix_train.T,design_matrix_train)
        tmp2=np.einsum("ji,ik->jk",design_matrix_train.T,binned_position_train)
        self.theta= np.einsum("ji,ik->j",inv(tmp1),tmp2)
        return self.theta
    def predict(self,design_matrix_test:np.array):
        '''
        predicting based on test data 
        '''
        return np.einsum("ij,j->i",design_matrix_test,self.theta)

if __name__=="__main__":
    import matplotlib.pyplot as plt

    all_data_dir=Path('Modules/data/alldata/')
    datalist=[x for x in all_data_dir.iterdir()]
    
    output_dir=Path("Output/data/linear_gaussian/")
    if not output_dir.exists():
        output_dir.mkdir()

    # -----load sample data
    # sample_data_index=1
    # data_dir=datalist[sample_data_index]
    # sample_name=str(data_dir).split('/')[-1]
    # sample_type = "CaMKII" if "CaMKII" in sample_name else "Control"
    # print(sample_name)

    for data_dir in tqdm(datalist):
        data_name=str(data_dir).split('/')[-1]
        position,spikes=data_loader(data_dir) # load data

        binned_position=bin_pos(position)
        time_bin_size=1/3 #second
        num_time_bins,num_cells = spikes.shape

        design_mat_all=design_matrix_decoder(spikes)

        # split traina and test
        n_time_bins_train=int(num_time_bins/2)

        design_mat_train, binned_position_train = design_mat_all[:n_time_bins_train] , binned_position[:n_time_bins_train].reshape(-1,1)
        design_mat_test, binned_position_test = design_mat_all[n_time_bins_train:] , binned_position[n_time_bins_train:].reshape(-1,1)

        lg=linear_gaussian()
        theta=lg.fit(design_mat_train, binned_position_train)
        prediction=lg.predict(design_mat_test)

        # save theta(parameter) , prediction , test_data
        with open(output_dir/(f"lg_predict_{data_name}.pickle"),"wb") as f:
            pickle.dump([theta,prediction,binned_position_test],f)

    
    # -----save sample theta(parameter) , prediction , test_data
    # with open(output_dir/(f"lg_predict_{sample_type}.pickle"),"wb") as f:
    #     pickle.dump([theta,prediction,binned_position_test],f)

