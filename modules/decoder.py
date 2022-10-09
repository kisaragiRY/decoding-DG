from os import stat
import numpy as np
from scipy.linalg import inv
from pathlib import Path
import pickle
from .func import *
from tqdm import tqdm
from scipy import stats

class LinearRegression():
    '''A linear guassian model.

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
        self.X_train=design_matrix_train
        self.y_train=binned_position_train
        self.penalty=penalty

        tmp1=np.einsum("ji,ik->jk",design_matrix_train.T,design_matrix_train)
        tmp2=np.einsum("ji,ik->jk",design_matrix_train.T,binned_position_train)
        self.theta= np.einsum("ji,ik->j",inv(tmp1+penalty*np.identity(len(tmp1))),tmp2)

    def predict(self,design_matrix_test:np.array):
        '''Predicting using fitted parameters based on test data.
        
        return the predicted results

        Parameter:
        ---------
        design_matrix_test: np.array
            test design matrix including one column full of 1 for the intercept

        '''
        self.X_test=design_matrix_test
        self.prediction=np.einsum("ij,j->i",self.X_test,self.theta)

class Results(RidgeRegression):
    """Contain RidgeRegression results.

    Including fitted parameters and hypothesis tests.
    """
    def __init__(self) -> None:
        super().__init__()

    def cal_overall_sig(self):
        """Run a hypothesis test for the overall coefficients in the model.

        The statistics=[(ESS-RSS)/(p-1)] / [RSS/(n-p)] ~ F-distribution(p-1,n-p),
        where ESS is explained sum of squares abd RSS is residual sum of squares.
        """
        n,p=self.X_train.shape
        # Residual Sum of Squares=y'y-theta_hat'X'y
        RSS=self.y_train.dot(self.y_train)-self.theta.dot(np.einsum("ji,ik->jk",self.X_train.T,self.y_train)) 
        # Explained sum of squares=∑(y_i-y_bar)
        ESS=np.sum(self.y_train-np.average(self.y_train))
        # Statistics
        F= ((ESS-RSS)/(p-1)) / (RSS/(n-p))
        # get p-value from F-distribution
        p_value=stats.f.sf(F,p-1,n-p)

        self.overall_sig= p_value

    def cal_indiv_sig(self):
        """Run a hypothesis test for individual coefficients 

        The statistics=t_i=theta_hat/(c_ii**.5 * sigma_hat) ~ t with n-p degree of freedom,
        wheret heta_hat is the fitted parameter, c_ii is the diagnal elements of inv(X'X), 
        sigma_hat**2=RSS/(n-p).
        If |t_i|>t(alpha/2), refuse hypothesis.
        """
        n,p=self.X_train.shape
        # Residual Sum of Squares=y'y-theta_hat'X'y
        RSS=self.y_train.dot(self.y_train)-self.theta.dot(np.einsum("ji,ik->jk",self.X_train.T,self.y_train)) 
        # inv(X'X)
        C=inv(np.einsum("ji,ik->jk",self.X_train.T,self.X_train))
        # sigma
        sigma2=RSS/(n-p)
        # list of t statistics for each element in theta_hat
        t_list=[self.theta[i]/(C[i,i]*sigma2**.5) for i in range(len(self.theta))]
        # p-value list based on the t_list
        p_value_list=[stats.t.cdf(t,n-p) for t in t_list]
        
        self.individual_sig= p_value_list


    def summary(self):
        """The summary of hypothesis tests
        """
        self.model={
            "model name": "Ridge Refression",
            "penalty":self.penalty,
            "fitted error":cal_mae(self.predict(self.X_train),self.y_train),
            "fitted parameter":self.theta,
            "overall sig":self.overall_sig,
            "individual sig":self.individual_sig,
            "prediction":self.prediction,
        }
        return self.model
