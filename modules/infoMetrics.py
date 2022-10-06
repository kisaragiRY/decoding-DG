import numpy as np
import pandas as pd

class InfoMetrics():
    """Calculate the mutual information between spikes and binned position.
    """
    def __init__(self,spikes:np.array,status:np.array) -> None:
        """Initialization.

        Parameter:
        ---------
        spikes: np.array
            neuorns spike count data
        status:np.array
            can be discretized position from continuous coordinates to discrete value 1,2,3...
            or discretized direction, discretized speed
        """
        self.spikes=spikes
        self.status=status
    
    def concat_data(self):
        """Concatenate status and spikes and return a dataframe
        """
        self.status_spikes=pd.DataFrame(np.hstack((self.status,self.spikes)))

    def cal_mutual_information(self)->float:
        """Calculate the mutual information between spikes and binned position.
        """
        self.concat_data()
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

if __name__=="__main__":
    #----------Calculate mutual information(MI) and multi multual information(MMI)----------

    from pathlib import Path
    from tqdm import tqdm
    from modules.func import *

    all_data_dir=Path('data/alldata/') # data directory
    datalist=[x for x in all_data_dir.iterdir()] # get the list of files under the data directory

    output_dir=Path("output/data/info_metrics/") # setup the output directory
    if not output_dir.exists():
        output_dir.mkdir()

    #----variables for patitioning the open filed
    partition_type="vertical"  # how to divide the open filed, possible options: vertical, horizontal, grid
    
    for data_dir in tqdm(datalist):
        data_name=str(data_dir).split('/')[-1]
        position,spikes=load_data(data_dir) # load data
        
        time_bin_size=1/3 #second
        num_time_bins,num_cells = spikes.shape

        I_list=[] # the information list for each n_parts choices
        for n_parts in range(3,21): # n_parts: how many parts be divided
            binned_position=bin_pos(position,n_parts,partition_type)
            info_metrics=InfoMetrics(spikes,binned_position)
            I_list.append(info_metrics.cal_mutual_information())



        
