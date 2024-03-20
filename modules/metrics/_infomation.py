from numpy.typing import NDArray

from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import njit, prange

from ..utils.util import nd_unique

@njit
def mutual_info(spikes: NDArray, status: NDArray, nueron_id: int) -> float:
        """Calculate the mutual information between spikes and binned position.
        """
        num_timepoints = len(spikes)
        spike = spikes[:, nueron_id]
        comb = np.vstack((status, spike)).T
        comb_uniques = nd_unique(comb)

        I=0
        for i in prange(len(comb_uniques)):
            row = comb_uniques[i]
            p_sk = sum([(i==row).all() for i in comb]) / num_timepoints
            p_s = sum((comb[:,0] == row[0])) / num_timepoints
            p_k = sum((comb[:,1] == row[1])) / num_timepoints

            if p_sk == 0:
                I+=0
            else:
                log = np.log2(p_sk / (p_s*p_k))
                I += p_sk * log

        return  I

@dataclass
class InfoMetrics:
    """Calculate the mutual information between spikes and binned position.

    Parameter:
    ---------
    spikes : NDArray
        neuorns spike count data

    status : NDArray
        discretized behaviors with values like 1,2,3...
        it can be discretized position, direction or speed.
    """
    spikes: NDArray
    status: NDArray
    def __post_init__(self) -> None:
        """Initialization.
        """
        self.num_timepoints = len(self.status)

    def cal_mi(self, nueron_id: int):
        """Calculate the mutual information between spikes and binned position.
        """
        return mutual_info(self.spikes, self.status, nueron_id)

    # def multi_mutu_info(self):
    #     '''Calculate the multi-mutual information.
    #     '''

    #     I=0
    #     for row in comb.drop_duplicates().iterrows():
    #         p_sk = comb[comb == row[1]].dropna().count()[0] / self.num_timepoints

    #         df_tem = comb[comb.iloc[:,1:] == row[1][1:]].iloc[:,1:].dropna() # index based on the spike count
    #         p_s_k = comb.loc[df_tem.index][comb.iloc[:,0] == row[1][0]].count()[0] / len(df_tem)
    #         p_s = comb[comb.iloc[:,0] == row[1][0]].count()[0]/self.num_timepoints
            
    #         if p_sk == 0:
    #             I += 0
    #         else:
    #             log = np.log2(p_s_k / p_s)
    #             I += p_sk * log

    #     return  I

if __name__ == "__main__":
    from pathlib import Path
    from dataloader import BaseDataset
    
    DATA_ROOT = Path('/work/data/processed/')
    data_list = np.array([x for x in DATA_ROOT.iterdir()])
    data_dir = data_list[2]

    dataset = BaseDataset(data_dir, 0.1, False, False)
    info = InfoMetrics(dataset.spikes, dataset._discretize_coords())
    [info.cal_mi(n) for n in range(dataset.spikes.shape[1])]
