from typing import Union, List
from numpy.typing import NDArray

from pathlib import Path
import numpy as np

from dataclasses import dataclass, field

@dataclass 
class ParamDir:
    """Parameters for directories."""
    ROOT : Path = Path("/work")
    DATA_ROOT : Path = ROOT/Path('data/processed/')
    OUTPUT_ROOT : Path = ROOT/Path("data/interim")

    def __post_init__(self) -> None:
        if not self.OUTPUT_ROOT.exists():
            self.OUTPUT_ROOT.mkdir()
        
        self.output_dir = self.OUTPUT_ROOT/"time_series_classification/"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.data_list = np.array([x for x in self.DATA_ROOT.iterdir()])

@dataclass
class ParamData:
    """Param for dataset.
    """
    window_size: int = 8
    train_ratio: float = .8
    mobility : float = 1.0
    K : int = 16 # the threshold for the length of segments
    shuffle: Union[bool, str] =  "segment label shuffling"  # there are two methods: behavior shuffling, events shuffling and segment label shuffling
    random_state : int = 202306
    stand_y_classes :  List = field(default_factory=lambda: [f"{i+1}" for i in range(4)])
    num_kernels_KO: int = 200
    num_kernels_WT: int = 200
    num_samples: int = 14

@dataclass
class ParamaRocketTrain:
    """Param for training.
    """
    model_name : str = "SVM" # "Ridge", "SVM"
    n_splits : int = 5 # for cross validation
    alphas: NDArray = np.logspace(-3, 3, 10)
    Cs: NDArray = np.logspace(-3, 3, 10)
    njobs: int = 30
    random_state : int = 202306
