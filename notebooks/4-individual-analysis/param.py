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
