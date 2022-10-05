import os
import pandas as pd
import numpy as np
from scipy.linalg import hankel
from pathlib import Path
 
def load_data(data_dir):
    '''
    load positiona and spike data
    '''
    position_df=pd.read_excel(data_dir/'position.xlsx')
    position=position_df.values[3:,1:3] # only take the X,Y axis data

    spikes_df=pd.read_excel(data_dir/'traces.xlsx',index_col=0)
    spikes=spikes_df.values

    # make sure spike and postion data have the same length
    n_bins=min(len(position),len(spikes))
    position = position[:n_bins]
    spikes = spikes[:n_bins]

    return position,spikes
    
def bin_pos(position,num_par=2,partition_type="grid"):
    '''
    discretize position, turn x,y axis positoin data from continuous number to 
    discretized number like 1,2,3...
    parameters:
    position: original position data which is in a range of [0 pixel,200 pixels]
    num_par: discretize position x-axis/y-axis into how many parts
    partition_type: the way to discretize data; vertical(II), horizontal(二), default:grid
    '''
    OF_size=200 # open field size
    bin_edges=np.linspace(0,OF_size,int(num_par)+1)

    if partition_type=="vertical":
        binned_position_x=np.digitize(position[:,0],bin_edges)
        return binned_position_x
    if partition_type=="horizontal":
        binned_position_y=np.digitize(position[:,1],bin_edges)
        return binned_position_y

    num_time_bins=len(position) # number of time bins

    binned_position_x=np.digitize(position[:,0],bin_edges)
    binned_position_y=np.digitize(position[:,1],bin_edges)

    binned_position=np.zeros(num_time_bins)
    for t in range(num_time_bins):
        x,y=binned_position_x[t],binned_position_y[t]
        binned_position[t]=(x-1)*num_par+y
    
    return binned_position


def mk_design_matrix_encoder(binned_position,spikes,ntfilt,nthist):
    '''
    to construct design matrix for GLM encoder

    parameters:
    ntfilt: number of time bins of position
    nthist: number of time bins of auto-regressive spike-history
    '''
    num_time_bins,num_cells = spikes.shape
    padded_pos = np.hstack((np.zeros(ntfilt-1), binned_position))   # pad early bins of stimulus with zero
    design_mat_pos = hankel(padded_pos[:-ntfilt+1], binned_position[-ntfilt:])

    design_mat_all_spikes = np.zeros((num_time_bins,nthist,num_cells)) 
    for j in np.arange(num_cells):
        padded_spikes = np.hstack((np.zeros(nthist), spikes[:-1,j]))
        design_mat_all_spikes[:,:,j] = hankel(padded_spikes[:-nthist+1], padded_spikes[-nthist:])

    # Reshape it to be a single matrix
    design_mat_all_spikes = np.reshape(design_mat_all_spikes, (num_time_bins,-1), order='F')
    design_mat_all = np.concatenate((design_mat_pos, design_mat_all_spikes), axis=1) 

    # add offfset
    design_mat_all_offset = np.hstack((np.ones((num_time_bins,1)), design_mat_all))

    return design_mat_all_offset

def mk_design_matrix_decoder(spikes:np.array,nthist:int=0):
    """Make design matrix with/without spike history for decoder.

    Parameter:
    ----------
    spikes: np.array
        that has neurons's spike counts data.
    nthist: int
        num of time bins for spike history,default=0
    """
    n_time_bins,n_neurons = spikes.shape
    design_mat_hist=np.zeros((n_time_bins,nthist,n_neurons))
    for neuron in range(n_neurons):
         design_mat_hist[:,:,neuron]=hankel(spikes[:-nthist+1,neuron],spikes[-nthist:,neuron])
    design_mat_hist= design_mat_hist.reshape(n_time_bins,-1,order='F')
    design_mat_all_offset = np.hstack((np.ones((n_time_bins,1)), design_mat_hist))
    return design_mat_all_offset
        


def cal_mse(prediction,observation):
    '''
    calculate the MSE based on prediction,observation
    '''
    tmp=[(i-j)**2 for i,j in zip(prediction,observation)]
    return np.sum(tmp)/len(prediction)

def cal_mae(prediction,observation):
    '''
    calculate the MAE based on prediction,observation
    '''
    if (prediction).all()==np.nan:
        tmp=np.nan
    else:
        tmp=[np.abs(i-j)for i,j in zip(prediction,observation)]
    return np.sum(tmp)/len(prediction)


if __name__=="__main__":
    """
    """
