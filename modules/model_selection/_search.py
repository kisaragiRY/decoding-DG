from dataclasses import dataclass
from typing import Union
from copy import deepcopy
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

from metrics import get_scorer
from decoder import RidgeRegression

@dataclass
class SearchCV:
    """Find the best params with cross validation."""
    estimator: RidgeRegression
    scoring: str
    candidate_params: list
    cv: TimeSeriesSplit

    def __post_init__(self) -> None:
        """Post processing."""
        self.scorer = get_scorer(self.scoring)

    def fit_and_score(self, X, y, train_index, test_index, hyper_param) -> dict:
        """Fit estimator and compute scores for a given dataset split."""
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        estimator = deepcopy(self.estimator)

        estimator.fit(X_train,y_train, hyper_param)
        fitted_param = estimator.fitted_param

        estimator.predict(X_test)

        y_pred = estimator.prediction

        result = {
            "train_scores": self.scorer(y_train, np.einsum("ij,j->i",X_train, fitted_param)),
            "test_scores" : self.scorer(y_test, y_pred),
            "fitted_parameters": fitted_param,
        }

        return result
    
    def evaluate_candidates(self, X, y, n_split):
        """Run search among cv splits and get the best parameters."""
        self.results = dict()
        min_score = np.inf
        for id_, param, (train_index, test_index) in enumerate(product(self.candidate_params,self.cv(n_split = n_split).split(X))):
            result = self.fit_and_score(X, y, train_index, test_index, param)

            # evaluate the the RMSE in one cv session !!

            self.results[id_] = {
                "fit_and_score": result,
                "hyper_param": param,
                }
            if result["test_scores"] <= min_score:
                min_score = result["test_scores"]
                self.best_result = self.results[id_]
                

    