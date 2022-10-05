import numpy as np
import pandas as pd

class InfoMetrics():
    """Calculate the mutual information between spikes and binned position.
    """
    def __init__(self,spikes:np.array,binned_position:np.array) -> None:
        """Initialization.

        Parameter:
        ---------
        spikes: np.array
            neuorns spike count data
        binned_position:np.array
            discretized position from continuous coordinates to discrete value 1,2,3...
        """
        self.spikes=spikes
        self.status=binned_position
    
    def concat_data(self):
        """Concatenate status and spikes and return a dataframe
        """
        self.status_spikes=pd.DataFrame(np.hstack(self.status,self.spikes))

    def cal_mutual_information(self)->float:
        """Calculate the mutual information between spikes and binned position.
        """

        time=len(self.status_spikes)
        I=0
        for row in self.status_spikes.drop_duplicates().iterrows():
            s=row[1][0] # status
            k=row[1][1] # spike
            p_sk=self.status_spikes[(self.status_spikes.iloc[:,0]==s) & (self.status_spikes.iloc[:,1]==k)].count()[0]/time
            p_s=self.status_spikes[(self.status_spikes.iloc[:,1]==k)].count()[0]/time
            p_k=self.status_spikes[(self.status_spikes.iloc[:,0]==s)].count()[0]/time

            log=np.log2(p_sk/(p_s*p_k))
            I+=p_sk*log

        return  I

    def multi_mutu_info(self):
        '''Calculate the multi-mutual information.
        '''

        time=len(self.status_spikes)
        I=0
        for row in self.status_spikes.drop_duplicates().iterrows():

            p_sk=self.status_spikes[self.status_spikes==row[1]].dropna().count()[0]/time

            df_tem=self.status_spikes[self.status_spikes.iloc[:,1:]==row[1][1:]].iloc[:,1:].dropna() # index based on the spike count
            p_s_k=self.status_spikes.loc[df_tem.index][self.status_spikes.iloc[:,0]==row[1][0]].count()[0]/len(df_tem)
            p_s=self.status_spikes[self.status_spikes.iloc[:,0]==row[1][0]].count()[0]/time
            
            if p_sk==0:
                I+=0
            else:
                log=np.log2(p_s_k/(p_s))
                I+=p_sk*log

        return  I