from scipy.linalg import inv
from scipy import stats
import numpy as np
from dataclasses import dataclass

from modules.decoder import RidgeRegression

@dataclass
class RidgeSigTest:
    model: RidgeRegression

    def __post_init__(self) -> None:
        """Post processing."""
        self._model_sig()
        self._coeff_sig()

    def _model_sig(self) -> None:
        """Run a hypothesis test for the model coefficients in the model.

        The statistics=[(ESS-RSS)/(p-1)] / [RSS/df_model] ~ F-distribution(p-1,df_model),
        where ESS is explained sum of squares abd RSS is residual sum of squares.
        """
        n, p = self.model.X_train.shape
        # Residual Sum of Squares = (y-Xw)'(y-Xw)
        RSS_tmp = self.model.y_train - np.einsum("ij,j->i", self.model.X_train , self.model.fitted_param)
        self.RSS = RSS_tmp.dot(RSS_tmp)

        # Explained sum of squares=∑(y_i-y_bar)^2
        ESS = np.sum((self.model.y_train - np.average(self.model.y_train))**2)

        # degree of freedom of the model
        self.C = inv(np.einsum("ji,ik->jk", self.model.X_train.T, self.model.X_train) + self.model.penalty * np.identity(p)) 
<<<<<<< HEAD
        H = np.einsum("ij,jk,kl -> il", self.model.X_train, self.C, self.model.X_train.T) # hat matrix
=======
        H_tmp = np.einsum("ji,ik->jk", self.model.X_train, self.C)
        H = np.einsum("ji,ik->jk", H_tmp, self.model.X_train.T)
>>>>>>> 1216e0547befbf579edaed9d3984dac5928e9795
        self.df_model = n - np.trace(2 * H - np.einsum("ij,jk -> ik", H, H.T))

        # Statistics
        self.f_stat = ((ESS - self.RSS) / (p-1)) / (self.RSS / self.df_model)

        # get p-value from F-distribution
        self.f_p_value = stats.f.sf(self.f_stat, p-1, self.df_model)

    def _coeff_sig(self) -> None:
        """Run a hypothesis test for coeff coefficients 

        The statistics=t_i=fitted_param_hat/(c_ii**.5 * sigma_hat) ~ t with n-p degree of freedom,
        wheret heta_hat is the fitted parameter, c_ii is the diagnal elements of inv(X'X), 
        sigma_hat**2=RSS/(n-p).
        If |t_i|>t(alpha/2), refuse hypothesis.
        """
        n, p = self.model.X_train.shape

        # estimate of sigma(sd of estimated coefficient)
        sigma = (self.RSS / self.df_model) ** .5

        # list of t statistics for each element in fitted_param_hat
        self.t_stat_list = [self.model.fitted_param[i] / (self.C[i,i] * sigma) for i in range(p)]

        # p-value list based on the t_list
        self.t_p_value_list = [stats.t.cdf(t, self.df_model) for t in self.t_stat_list]
