from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import pandas as pd

@dataclass
class InfoMetrics:
    """Calculate the mutual information between spikes and binned position.

    Parameter:
    ---------
    spikes: NDArray
        neuorns spike count data

    status:NDArray
        can be discretized position from continuous coordinates to discrete value 1,2,3...
        or discretized direction, discretized speed
    """
    spikes: NDArray
    status: NDArray
    def __post_init__(self) -> None:
        """Initialization.
        """
        self.status = self.status.reshape(-1,1)
    
    def concat_data(self, nueron_id: int):
        """Concatenate status and spikes and return a dataframe
        """
        self.status_spikes=pd.DataFrame(np.hstack((self.status,self.spikes[:,nueron_id].reshape(-1,1))))

    def mutual_info(self, nueron_id: int) -> float:
        """Calculate the mutual information between spikes and binned position.
        """
        self.concat_data(nueron_id)
        time = len(self.status_spikes)
        I=0
        for row in self.status_spikes.drop_duplicates().iterrows():
            s = row[1][0] # status
            k = row[1][1] # spike
            p_sk = self.status_spikes[(self.status_spikes.iloc[:,0] == s) & (self.status_spikes.iloc[:,1] == k)].count()[0] / time
            p_s = self.status_spikes[(self.status_spikes.iloc[:,1] == k)].count()[0] / time
            p_k = self.status_spikes[(self.status_spikes.iloc[:,0] == s)].count()[0] / time

            if p_sk == 0:
                I+=0
            else:
                log = np.log2(p_sk / (p_s*p_k))
                I += p_sk * log

        return  I

    def multi_mutu_info(self):
        '''Calculate the multi-mutual information.
        '''

        time = len(self.status_spikes)
        I=0
        for row in self.status_spikes.drop_duplicates().iterrows():
            p_sk = self.status_spikes[self.status_spikes == row[1]].dropna().count()[0] / time

            df_tem = self.status_spikes[self.status_spikes.iloc[:,1:] == row[1][1:]].iloc[:,1:].dropna() # index based on the spike count
            p_s_k = self.status_spikes.loc[df_tem.index][self.status_spikes.iloc[:,0] == row[1][0]].count()[0] / len(df_tem)
            p_s = self.status_spikes[self.status_spikes.iloc[:,0] == row[1][0]].count()[0]/time
            
            if p_sk == 0:
                I += 0
            else:
                log = np.log2(p_s_k / p_s)
                I += p_sk * log

        return  I