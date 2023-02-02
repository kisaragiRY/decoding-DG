from typing import List

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

        self.output_dir = self.OUTPUT_ROOT/"ridge_regression/"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.data_path_list = np.array([x for x in self.DATA_ROOT.iterdir()])

@dataclass
class ParamPastCoordData:
    """Param for setting up the dataset with past coordinates."""
    nthist_range: range = range(6)
    coord_axis_opts: List = field(default_factory=["x-axis","y-axis"])
    window_size_range: range = range(1, 30)

@dataclass
class ParamFiringRateData:
    """Param for setting up the dataset with only firing rate."""
    nthist: int = 0
    coord_axis_opts: List[str] = field(default_factory = lambda: ["x-axis","y-axis"])
    window_size_range: List[int] = field(default_factory = lambda: [int(2**i) for i in range(2, 9)])
    sigma_range: List[float] = field(default_factory = lambda: [2**i*.01 for i in range(7)])

@dataclass
class ParamTrain:
    """Param for training model."""
    # range of nthist(number of time bins for history)
    n_split : int = 5
    penalty_range = np.arange(.1, 1.1,.05)
    scoring = "mean_square_error"
    train_size = .8

@dataclass
class ParamEval:
    """Param for evaluating model."""
    scoring = "mean_square_error"