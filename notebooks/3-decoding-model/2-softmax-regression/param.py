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

        self.output_dir = self.OUTPUT_ROOT/"softmax_regression/"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.data_path_list = np.array([x for x in self.DATA_ROOT.iterdir()])

@dataclass
class ParamData:
    """Param for dataset.
    """
    window_size : float = 5
    train_ratio: float = .8

@dataclass
class ParamTrain:
    """Param for training.
    """
    lr: float = .005
    max_iter : int = 3000
