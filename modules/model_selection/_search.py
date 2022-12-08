from dataclasses import dataclass
from copy import deepcopy
from itertools import product
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from metrics import get_scorer
from decoder import RidgeRegression
from ._split import RollingOriginSplit
from ._sigtest import RidgeSigTest

@dataclass
class SearchCV:
    """Find the best params with cross validation."""
    scoring: str
    candidate_params: list
    n_split : int

    def __post_init__(self) -> None:
        """Post processing."""
        self.scorer = get_scorer(self.scoring)
        self.cv = RollingOriginSplit(self.n_split)
        self.out = None

    def fit_and_score(self, estimator: RidgeRegression, X: np.array, y: np.array, train_indexes: range, test_indexes: range, hyper_param: float) -> dict:
        """Fit estimator and compute scores for a given dataset split."""
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

        estimator.fit(X_train,y_train, hyper_param)
        fitted_param = estimator.fitted_param

        estimator.predict(X_test)
        y_pred = estimator.prediction

        result = {
            "train_scores": self.scorer(y_train, np.einsum("ij,j->i",X_train, fitted_param)),
            "test_scores" : self.scorer(y_test, y_pred),
            "estimator" : estimator
        }
        return result
    
    def evaluate_candidates(self, X, y):
        """Run search among cv splits and get the best parameters."""

        parallel = Parallel(n_jobs=-1)
        self.out = parallel(delayed(self.fit_and_score)(
                    RidgeRegression(),
                    X,
                    y,
                    train_indexes, 
                    test_indexes, 
                    param
                ) 
                for param, (train_indexes, test_indexes) in 
                product(self.candidate_params, self.cv.split(X)))

    def _aggregate_result(self):
        """Aggregate results to a dict."""
        agg_out = {key: [result[key] for result in self.out] for key in self.out[0]}
        return agg_out
    
    @property
    def best_result(self):
        """Get the best result with lowest test_scores
        
        Return
        ----------
        train_scores: scores during cross-validation training.
        test_scores : scores during cross-validation testing.
        estimator: estimator instance.
        sig_tests: significance tests results.
        """
        if self.out is None:
            raise ValueError("evaluate_candidates is not implemented.")
        else:
            agg_out = self._aggregate_result()
            best_index = np.argmin(agg_out["test_scores"])
            best_result = self.out[best_index]

            sig_tests = RidgeSigTest(best_result["estimator"])
            more_results ={
                "sig_tests": sig_tests,
            }
            best_result.update(more_results)

        return best_result
        


                