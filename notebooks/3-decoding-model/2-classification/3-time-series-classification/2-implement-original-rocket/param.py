from typing import List
from numpy.typing import NDArray

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class ParamDir:
    """Param for directory."""
    ROOT : Path = Path("/work")
    DATA_ROOT : Path = ROOT/Path('data/alldata/')
    OUTPUT_ROOT : Path = ROOT/Path("data/interim")

    def __post_init__(self) -> None:
        if not self.OUTPUT_ROOT.exists():
            self.OUTPUT_ROOT.mkdir()

        self.output_dir = self.OUTPUT_ROOT/"time_series_classification/"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.data_path_list = np.array([x for x in self.DATA_ROOT.iterdir()])

@dataclass
class ParamData:
    """Param for dataset.
    """
    window_size: int = 8
    train_ratio: float = .8
    mobility : float = 1.0
    K : int = 16 # the threshold for the length of segments
    shuffle: bool =  False # there are two methods: behavior shuffling and events shuffling
    reduction_method: str = "LEM"
    random_state : int = 20230411
    num_kernels_KO: int = 250
    num_kernels_WT: int = 300

@dataclass
class ParamaRocketTrain:
    """Param for training.
    """
    model_name : str = "Ridge" # "Ridge", "SVM", "Softmax", 
    n_splits : int = 5 # for cross validation
    alphas: NDArray = np.logspace(-3, 3, 10)
    Cs: NDArray = np.logspace(-3, 3, 10)
    njobs: int = 30
