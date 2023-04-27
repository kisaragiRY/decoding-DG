from typing import Union

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class ParamDir:
    """Param for directory."""
    ROOT : Path = Path("/work")
    DATA_ROOT : Path = ROOT/Path('data/processed')
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
    """Params for dataset.
    """
    kernel_dim : int = 2
    window_size = 8
    mobility : float = 1.0
    shuffle =  False # there are two methods: behavior shuffling and events shuffling
    random_state : Union[int, bool] = 202304

@dataclass
class ParamTrain:
    """Param for training.
    """
    model_name : str = "spectral" # available options: "k_means", "spectral"
    n_clusters : int = 4
    assign_labels : str = "cluster_qr"
    random_state : Union[int, bool] = 202304