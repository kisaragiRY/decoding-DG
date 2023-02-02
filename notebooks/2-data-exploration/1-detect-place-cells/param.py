from pathlib import Path
import numpy as np

from dataclasses import dataclass

@dataclass 
class ParamDir:
    """Parameters for directories."""
    ROOT : Path = Path("/work")
    DATA_ROOT : Path = ROOT/Path('data/alldata/')
    OUTPUT_ROOT : Path = ROOT/Path("data/interim")

    def __post_init__(self) -> None:
        if not self.OUTPUT_ROOT.exists():
            self.OUTPUT_ROOT.mkdir()

        self.output_dir = self.OUTPUT_ROOT/ "data_exploration/"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.data_list = np.array([x for x in self.DATA_ROOT.iterdir()])

@dataclass
class ParamShuffle:
    """Parameters for shuffling."""
    num_repeat : int = 500
