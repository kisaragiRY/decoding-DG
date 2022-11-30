from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ParamDir:
    """Param for directory."""
    DATA_DIR : Path = Path('data/alldata/')
    OUTPUT_DIR : Path = Path("output/data/ridge_regression/")

    def __post_init__(self) -> None:
        self.data_path_list=[x for x in self.DATA_DIR.iterdir()]

@dataclass
class ParamData:
    """Param for setting up dataset."""
    nthist_range = np.arange(0,18,3)

@dataclass
class ParamTrain:
    """Param for training model."""
    # range of nthist(number of time bins for history)
    penalty_range = np.arange(4,18,.2)
    scoring = "mean_sqaure_error"