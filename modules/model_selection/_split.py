from typing import Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class RollingOriginSplit:
    """Get validation data from X.
    
    Based on rolling origin cross validation techniques,
    one fold from n-fold of data would be used for validating,
    and the rest would be used to training.

    Parameter
    ---------
    train_data: Tuple[np.array, np.array]
        the training data that includes X_train and y_train
    n_split: int
        the number of folds to split the data
    
    Return
    ---------
    train_indexes: range
        the indexes for train set
    test_indexes: range
        the indexes for  test set
    """
    n_split: int = 5

    def __post_init__(self) -> None:
        pass

    def split(self, X) -> Tuple[Tuple[np.array, np.array], Tuple[range, range]]:
        fold_size = int( len(X) / self.n_split )
        for id_fold in range(self.n_split):
            id_index = ( id_fold + 1 ) * fold_size
            train_indexes, test_indexes = range(id_index-1), range(id_index-1, id_index)
            yield  train_indexes, test_indexes