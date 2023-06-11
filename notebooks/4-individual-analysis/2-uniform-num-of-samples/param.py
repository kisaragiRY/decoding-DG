from typing import Union
from pathlib import Path
import numpy as np

from dataclasses import dataclass

@dataclass 
class ParamDir:
    """Parameters for directories."""
    ROOT : Path = Path("/work")
    DATA_ROOT : Path = ROOT/Path('data/processed/')
    OUTPUT_ROOT : Path = ROOT/Path("data/interim")

    def __post_init__(self) -> None:
        if not self.OUTPUT_ROOT.exists():
            self.OUTPUT_ROOT.mkdir()

        self.data_list = np.array([x for x in self.DATA_ROOT.iterdir()])

@dataclass
class ParamData:
    """Param for dataset.
    """
    window_size: int = 8
    train_ratio: float = .8
    mobility : float = 1.0
    K : int = 16 # the threshold for the length of segments
    shuffle: Union[bool, str] =  False # there are two methods: behavior shuffling, events shuffling and segment label shuffling
    random_state : int = 202304