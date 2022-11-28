from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ParamDir:
    """Param for directory."""
    DATA_DIR : Path = Path('data/alldata/')
    OUTPUT_DIR : Path = Path("output/ridge_regression/")

    def __post_init__(self) -> None:
        self.data_path_list = np.array([x for x in self.DATA_DIR.iterdir()])

@dataclass
class ParamData:
    """Param for setting up dataset."""
    nthist_range = [0,1]

@dataclass
class ParamTrain:
    """Param for training model."""
    # range of nthist(number of time bins for history)
    n_split : int = 5
    penalty_range = np.arange(1,4,.5)
    scoring = "mean_square_error"
    train_size = .8

@dataclass
class ParamEval:
    """Param for evaluating model."""
    scoring = "mean_square_error"