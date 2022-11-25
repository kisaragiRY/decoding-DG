from dataclasses import dataclass
from functools import cached_property
import numpy as np
from scipy.linalg import inv
from .func import *
from scipy import stats

class RidgeRegression():
    '''A linear guassian ridge model.

    x_t=fitted_param.T・n_t + b_t
    x_t: discretized position
    fitted_param: parameter
    n_t: spikes
    b_t: intercept
    '''
    def __init__(self) -> None:
        pass

    def fit(self,X_train:np.array,y_train:np.array,penalty:float):
        '''Fitting based on training data.
        
        return the fitted coefficients

        Parameter:
        ---------
        X_train: np.array
            train design matrix including one column full of 1 for the intercept
        y_train: np.array
            discretized position from continuous coordinates to discrete value 1,2,3...
        penalty: float
            the penalty added on ridge model
        '''
        self.X_train=X_train
        self.y_train=y_train.ravel()
        self.penalty=penalty

        tmp1=np.einsum("ji,ik->jk",self.X_train.T,self.X_train)
        tmp2=np.einsum("ji,i->j",self.X_train.T,self.y_train)
        try: 
            inv(tmp1+penalty*np.identity(len(tmp1)))
            self.fitted_param = np.einsum("ji,i->j",inv(tmp1+penalty*np.identity(len(tmp1))),tmp2)
            self.fitting=True # indicate whether the fitting is successfully conducted

        except: 
            self.fitting=False
            self.fitted_param= np.array([np.nan]*self.X_train.shape[1])

    def predict(self, X_test: np.array):
        '''Predicting using fitted parameters based on test data.
        
        return the predicted results

        Parameter:
        ---------
        design_matrix_test: np.array
            test design matrix including one column full of 1 for the intercept

        '''
        self.prediction = np.einsum("ji,i->j", X_test, self.fitted_param)

@dataclass
class Results:
    """Contain RidgeRegression results.

    Including fitted parameters and hypothesis tests.
    """
    model:RidgeRegression

    @cached_property
    def overall_sig(self):
        """Run a hypothesis test for the overall coefficients in the model.

        The statistics=[(ESS-RSS)/(p-1)] / [RSS/(n-p)] ~ F-distribution(p-1,n-p),
        where ESS is explained sum of squares abd RSS is residual sum of squares.
        """
        n,p=self.model.X_train.shape
        # Residual Sum of Squares=y'y-fitted_param_hat'X'y
        RSS = self.model.y_train.dot(self.model.y_train) - self.model.fitted_param.dot(np.einsum("ji,ik->jk",self.model.X_train.T,self.model.y_train)) 
        # Explained sum of squares=∑(y_i-y_bar)
        ESS=np.sum(self.model.y_train-np.average(self.model.y_train))
        # Statistics
        F= ((ESS-RSS)/(p-1)) / (RSS/(n-p))
        # get p-value from F-distribution
        p_value=stats.f.sf(F,p-1,n-p)

        return np.array(p_value).ravel()

    @cached_property
    def individual_sig(self):
        """Run a hypothesis test for individual coefficients 

        The statistics=t_i=fitted_param_hat/(c_ii**.5 * sigma_hat) ~ t with n-p degree of freedom,
        wheret heta_hat is the fitted parameter, c_ii is the diagnal elements of inv(X'X), 
        sigma_hat**2=RSS/(n-p).
        If |t_i|>t(alpha/2), refuse hypothesis.
        """
        n,p=self.model.X_train.shape
        # Residual Sum of Squares=y'y-fitted_param_hat'X'y
        RSS = self.model.y_train.dot(self.model.y_train) - self.model.fitted_param.dot(np.einsum("ji,i->j",self.model.X_train.T,self.model.y_train)) 

        try: 
            inv_tmp=inv(np.einsum("ji,ik->jk",self.model.X_train.T,self.model.X_train) + self.model.penalty*np.ones(p)) # (X'X+λI)^-1
            tmp1=np.einsum("ji,ik->jk",inv_tmp,self.model.X_train.T) # inv_tmp@X'
            tmp2=np.einsum("ji,ik->jk",self.model.X_train,inv_tmp) # X@inv_tmp
            C=np.einsum("ji,ik->jk",tmp1,tmp2)
        except:
            C=np.empty((p,p))
            C[:]=np.nan

        # sigma
        sigma2=RSS/(n-p)
        # list of t statistics for each element in fitted_param_hat
        t_list=[self.model.fitted_param[i]/(C[i,i]*sigma2**.5) for i in range(len(self.model.fitted_param))]
        # p-value list based on the t_list
        p_value_list=[stats.t.cdf(t,n-p) for t in t_list]
        
        return np.array(p_value_list).ravel()


    def summary(self):
        """The summary of hypothesis tests
        """
        smry={
            "model name": "Ridge Regression",
            "fitting":self.model.fitting,
            "penalty":self.model.penalty,
            "fitted parameter":self.model.fitted_param,
            "fitted error":cal_mse(np.einsum("ij,j->i",self.model.X_train,self.model.fitted_param),self.model.y_train),
            "overall sig":self.overall_sig,
            "individual sig":self.individual_sig,
            "prediction":self.model.prediction,
        }
        return smry
