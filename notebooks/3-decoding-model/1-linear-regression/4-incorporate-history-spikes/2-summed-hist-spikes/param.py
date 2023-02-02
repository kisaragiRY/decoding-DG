from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ParamDir:
    """Param for directory."""
    ROOT : Path = Path("/work")
    DATA_ROOT : Path = ROOT/Path('data/alldata/')
    OUTPUT_ROOT : Path = ROOT/Path("output/")

    def __post_init__(self) -> None:
        if not self.OUTPUT_ROOT.exists():
            self.OUTPUT_ROOT.mkdir()

        self.output_dir = self.OUTPUT_ROOT/ "ridge_regression/"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.data_path_list = np.array([x for x in self.DATA_ROOT.iterdir()])

@dataclass
class ParamData:
    """Param for setting up the dataset."""
    nthist_range = range(6)
    coord_axis_opts = ["x-axis","y-axis"]
    window_size_range = range(1, 30)
    mode = "summed-past"

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