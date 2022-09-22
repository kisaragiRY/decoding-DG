import numpy as np
from scipy.linalg import inv

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
        self.theta= np.einsum("ji,ik->jk",inv(tmp1),tmp2)
        return self.theta
    def predict(self,design_matrix_test:np.array):
        '''
        predicting based on test data 
        '''
        return np.einsum("ji,ik->jk",self.theta,design_matrix_test)



