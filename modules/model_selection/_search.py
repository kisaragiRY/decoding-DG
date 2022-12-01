from dataclasses import dataclass
from copy import deepcopy
from itertools import product
import numpy as np
from tqdm import tqdm
<<<<<<< HEAD
=======
from joblib import Parallel, delayed
>>>>>>> 1216e0547befbf579edaed9d3984dac5928e9795

from modules.metrics import get_scorer
from modules.decoder import RidgeRegression
from ._split import RollingOriginSplit
from ._sigtest import RidgeSigTest

@dataclass
class SearchCV:
    """Find the best params with cross validation."""
    estimator: RidgeRegression
    scoring: str
    candidate_params: list
    n_split : int

    def __post_init__(self) -> None:
        """Post processing."""
        self.scorer = get_scorer(self.scoring)
        self.cv = RollingOriginSplit(self.n_split)

<<<<<<< HEAD
    def fit_and_score(self, X, y, train_indexes, test_indexes, hyper_param) -> dict:
=======
    def fit_and_score(self, estimator: RidgeRegression, X: np.array, y: np.array, train_indexes: range, test_indexes: range, hyper_param: float) -> dict:
>>>>>>> 1216e0547befbf579edaed9d3984dac5928e9795
        """Fit estimator and compute scores for a given dataset split."""
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

<<<<<<< HEAD
        estimator = deepcopy(self.estimator)

=======
>>>>>>> 1216e0547befbf579edaed9d3984dac5928e9795
        estimator.fit(X_train,y_train, hyper_param)
        fitted_param = estimator.fitted_param

        estimator.predict(X_test)
        y_pred = estimator.prediction

        sig_tests = RidgeSigTest(estimator)

        result = {
            "train_scores": self.scorer(y_train, np.einsum("ij,j->i",X_train, fitted_param)),
            "test_scores" : self.scorer(y_test, y_pred),
            "fitted_param": fitted_param,
            "hyper_param": hyper_param,
            "RSS": sig_tests.RSS,
            "F_stat": sig_tests.f_stat,
            "F_p_value": sig_tests.f_p_value,
            "coeff_stats": sig_tests.t_stat_list,
            "coeff_p_values": sig_tests.t_p_value_list
        }
<<<<<<< HEAD

=======
>>>>>>> 1216e0547befbf579edaed9d3984dac5928e9795
        return result
    
    def evaluate_candidates(self, X, y):
        """Run search among cv splits and get the best parameters."""
<<<<<<< HEAD
        self.results = dict()
        min_score = np.inf
        for id_, (param, (train_indexes, test_indexes)) in tqdm(enumerate(product(self.candidate_params,self.cv.split(X)))):
            result = self.fit_and_score(X, y, train_indexes, test_indexes, param)

            self.results[id_] = result

            # get the best result with lowest test_scores
            if result["test_scores"] <= min_score:
                min_score = result["test_scores"]
                self.best_result = result
=======

        parallel = Parallel(n_jobs=-1)
        self.out = parallel(delayed(self.fit_and_score)(
                    train_indexes, 
                    test_indexes, 
                    param
                ) 
                for param, (train_indexes, test_indexes) in 
                tqdm(product(self.candidate_params, self.cv(X))))

    def _aggregate_result(self):
        """Aggregate results to a dict."""
        agg_out = {key: [result[key] for result in self.out] for key in self.out[0]}
        return agg_out
    
    @property
    def best_result(self):
        """Get the best result with lowest test_scores"""
        agg_out = self._aggregate_result(self.out)
        best_index = agg_out["test_scores"].argmin()
        return self.out[best_index]
        


>>>>>>> 1216e0547befbf579edaed9d3984dac5928e9795
                