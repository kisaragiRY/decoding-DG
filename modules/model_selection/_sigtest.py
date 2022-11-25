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

        The statistics=[(ESS-RSS)/(p-1)] / [RSS/(n-p)] ~ F-distribution(p-1,n-p),
        where ESS is explained sum of squares abd RSS is residual sum of squares.
        """
        n, p = self.model.X_train.shape
        # Residual Sum of Squares=y'y-fitted_param_hat'X'y
        RSS = self.model.y_train.dot(self.model.y_train) - self.model.fitted_param.dot(np.einsum("ji,i->j ", self.model.X_train.T, self.model.y_train)) 
        # Explained sum of squares=∑(y_i-y_bar)
        ESS = np.sum(self.model.y_train - np.average(self.model.y_train))
        # Statistics
        self.f_stat = ((ESS - RSS) / (p-1)) / (RSS / (n-p))
        # get p-value from F-distribution
        self.f_p_value = stats.f.sf(self.f_stat,p-1,n-p)

    def _coeff_sig(self) -> None:
        """Run a hypothesis test for coeff coefficients 

        The statistics=t_i=fitted_param_hat/(c_ii**.5 * sigma_hat) ~ t with n-p degree of freedom,
        wheret heta_hat is the fitted parameter, c_ii is the diagnal elements of inv(X'X), 
        sigma_hat**2=RSS/(n-p).
        If |t_i|>t(alpha/2), refuse hypothesis.
        """
        n, p = self.model.X_train.shape
        # Residual Sum of Squares=y'y-fitted_param_hat'X'y
        RSS = self.model.y_train.dot(self.model.y_train) - self.model.fitted_param.dot(np.einsum("ji,i->j", self.model.X_train.T, self.model.y_train)) 

        try: 
            inv_tmp = inv(np.einsum("ji,ik->jk", self.model.X_train.T, self.model.X_train) + self.model.penalty * np.ones(p)) # (X'X+λI)^-1
            tmp1 = np.einsum("ji,ik->jk", inv_tmp, self.model.X_train.T) # inv_tmp@X'
            tmp2 = np.einsum("ji,ik->jk", self.model.X_train, inv_tmp) # X@inv_tmp
            C = np.einsum("ji,ik->jk", tmp1, tmp2)
        except:
            C = np.empty((p, p))
            C[:] = np.nan

        # sigma
        sigma2 = RSS / (n-p)
        # list of t statistics for each element in fitted_param_hat
        self.t_stat_list = [self.model.fitted_param[i] / (C[i,i] * sigma2 ** .5) for i in range(len(self.model.fitted_param))]
        # p-value list based on the t_list
        self.t_p_value_list = [stats.t.cdf(t,n-p) for t in self.t_stat_list]

if __name__ == "__main__":
    from _search import SearchCV
    from modules.metrics import get_scorer

    n_neurons=30
    time_bins_train=100
    time_bins_test=80
    n_positions=4
    X = np.random.rand(time_bins_train,n_neurons)
    y = np.random.uniform(low=-40, high=40, size=(time_bins_train,1))
    train_index, test_index, hyper_param = range(30), range(30,31), 5

    search = SearchCV(RidgeRegression(), "mean_square_error", np.arange(4,18,.2), 10)
    result = search.fit_and_score(X, y, train_index, test_index, hyper_param)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()

    rr = RidgeRegression()
    rr.fit(X_train, y_train, hyper_param)
    rr.predict(X_test)

    scorer = get_scorer("mean_square_error")
    train_scores = scorer(y_train, np.einsum("ij,j->i",X_train, rr.fitted_param))
    test_scores = scorer(y_test, rr.prediction)