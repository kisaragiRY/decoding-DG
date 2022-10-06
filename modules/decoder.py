from turtle import pen
import numpy as np
from scipy.linalg import inv
from pathlib import Path
import pickle
from func import *
from tqdm import tqdm

class LinearRegression():
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
        return the fitted coefficients
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

class RidgeRegression():
    '''A linear guassian ridge model.

    x_t=theta.T・n_t + b_t
    x_t: discretized position
    theta: parameter
    n_t: spikes
    b_t: intercept
    '''
    def __init__(self) -> None:
        pass

    def fit(self,design_matrix_train:np.array,binned_position_train:np.array,penalty:float):
        '''Fitting based on training data.
        
        return the fitted coefficients

        Parameter:
        ---------
        design_matrix_train: np.array
            train design matrix including one column full of 1 for the intercept
        binned_position_train: np.array
            discretized position from continuous coordinates to discrete value 1,2,3...
        penalty: float
            the penalty added on ridge model
        '''
        tmp1=np.einsum("ji,ik->jk",design_matrix_train.T,design_matrix_train)
        tmp2=np.einsum("ji,ik->jk",design_matrix_train.T,binned_position_train)
        self.theta= np.einsum("ji,ik->j",inv(tmp1+penalty*np.identity(len(tmp1))),tmp2)
        return self.theta

    def results(self):
        """Contain RidgeRegression results.

        Including fitted parameters and hypothesis tests.
        """
        def overall_sig(self):
            """Run a hypothesis test for the overall coefficients in the model.
            """
        def indiv_sig(self):
            """Run a hypothesis test for individual coefficients 
            """
        


    def predict(self,design_matrix_test:np.array):
        '''Predicting using fitted parameters based on test data.
        
        return the predicted results

        Parameter:
        ---------
        design_matrix_test: np.array
            test design matrix including one column full of 1 for the intercept

        '''
        return np.einsum("ij,j->i",design_matrix_test,self.theta)

if __name__=="__main__":
    import matplotlib.pyplot as plt

    #----variables
    decoder_m="ridge regression" # decoder method
    partition_type="vertical"
    n_parts=4

    all_data_dir=Path('data/alldata/')
    datalist=[x for x in all_data_dir.iterdir()]
    
    # output_dir=Path("Output/data/linear_gaussian/")
    output_dir=Path("output/data/linear_gaussian_ridge/")
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
        position,spikes=load_data(data_dir) # load data

        # binned_position=bin_pos(position,n_parts,partition_type)
        binned_position=position
        time_bin_size=1/3 #second
        num_time_bins,num_cells = spikes.shape

        design_mat_all=mk_design_matrix_decoder(spikes)

        # split traina and test
        n_time_bins_train=int(num_time_bins/2)

        design_mat_train, binned_position_train = design_mat_all[:n_time_bins_train] , binned_position[:n_time_bins_train].reshape(-1,1)
        design_mat_test, binned_position_test = design_mat_all[n_time_bins_train:] , binned_position[n_time_bins_train:].reshape(-1,1)

        if decoder_m=="linear regression": # ----this is only for two samples data(control & camkII)
            sample_name=data_name
            sample_type = "CaMKII" if "CaMKII" in sample_name else "Control"
            lg=LinearRegression()
            try: 
                theta=lg.fit(design_mat_train, binned_position_train)
                prediction=lg.predict(design_mat_test)
            except:
                print("fitting failed")
                theta=[np.nan]*design_mat_train.shape[1]
                prediction=[np.nan]*len(binned_position_test)
            # -----save sample theta(parameter) , prediction , test_data
            with open(output_dir/(f"lg_predict_{sample_type}.pickle"),"wb") as f:
                pickle.dump([theta,prediction,binned_position_test],f)

        elif decoder_m=="ridge regression":
            theta_prediction_penalty=[]
            failed_penalty=[]
            # for p in range(10):
            for p in [2**i for i in range(3,13)]:
                lgr=RidgeRegression()
                try: 
                    theta=lgr.fit(design_mat_train, binned_position_train,p)
                    prediction=lgr.predict(design_mat_test)
                    prediction_train=lgr.predict(design_mat_train)
                except:
                    print("fitting failed")
                    failed_penalty.append(p)
                    # if fitting failed, set the following variables to np.nan
                    theta=np.array([np.nan]*design_mat_train.shape[1])
                    prediction=np.array([np.nan]*len(binned_position_test))
                    prediction_train=np.array([np.nan]*len(binned_position_test))
                theta_prediction_penalty.append([theta,prediction,prediction_train,p])

            # save theta(parameter) , prediction , test_data
            # with open(output_dir/(f"lgr_predict_{data_name}.pickle"),"wb") as f:
            # with open(output_dir/(f"lgr_predict_{data_name}_withLargerPenalty_{n_parts}_{partition_type}.pickle"),"wb") as f:
            with open(output_dir/(f"lgr_predict_{data_name}_withoutPartition.pickle"),"wb") as f:
                pickle.dump([theta_prediction_penalty,binned_position_test,binned_position_train,failed_penalty],f)

    

