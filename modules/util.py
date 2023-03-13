from typing import Tuple
from numpy.typing import NDArray

import numpy as np
from numpy.linalg import det, inv
from sklearn.utils import resample


def gauss1d(xx: NDArray, mu: float = 0, sigma: float = .2):
    """A Gaussian kernel."""
    kernel = 1 / ((2 * np.pi) ** 2 * sigma) * np.exp(- (xx - mu) ** 2 / (2 * sigma ** 2))
    return kernel / sum(kernel)

def gauss2d(xx: NDArray, mu: float = 0, sigma: float = 3):
    """A 2 dimension Gaussian kernel."""
    kenel1d = gauss1d(xx, mu, sigma)
    return kenel1d[:, np.newaxis] * kenel1d[np.newaxis, :]

def bin_pos(coords: NDArray, num_par: int = 2, partition_type : str = "grid") -> NDArray[np.int_]:
    """Discretize coordinates.
    
    turn x,y axis coodinates data from continuous number to 
    discretized number like 1,2,3...

    Parameters
    -----------
    coords: NDArray
        original coordinates data which is in a range of [0 pixel,200 pixels]
    num_par: int = 2
        discretize coord x-axis/y-axis into how many parts
    partition_type: : str = "grid"
        the method to discretize data; vertical(II), horizontal(二), default:grid
    """
    if partition_type not in ["grid", "vertical", "horizontal"]:
        raise ValueError("partition_type is invalid. It can be either 'grid', 'vertical' or 'horizontal'.")

    OF_size = 40 # open field size
    bin_edges = np.linspace(0, OF_size, int(num_par) + 1)
    actual_coord = coords * (OF_size / 200.0) 

    if partition_type == "vertical":
        binned_position_x = np.digitize(actual_coord[:,0], bin_edges)
        return binned_position_x
    if partition_type == "horizontal":
        binned_position_y = np.digitize(actual_coord[:,1], bin_edges)
        return binned_position_y

    num_time_bins = len(actual_coord) # number of time bins

    binned_position_x = np.digitize(actual_coord[:,0], bin_edges)
    binned_position_y = np.digitize(actual_coord[:,1], bin_edges)

    binned_position = np.zeros(num_time_bins)
    for t in range(num_time_bins):
        x, y = binned_position_x[t], binned_position_y[t]
        binned_position[t] = int((y-1) * num_par + x)
    
    return binned_position

def cal_velocity(coord: NDArray) -> NDArray:
    """Calculate the velocity from coordinates. """
    lenposi = len(coord)
    v_cm_s = []
    actual_posi = coord * (40.0 / 200.0) 
    for k in range(lenposi):
        tmp=0
        for i in range(-4,3): #313 filter
            if (k+i <= 0 or k+i+1 >= lenposi):
                tmp = tmp
            else:
                xaya = actual_posi[k+i+1] - actual_posi[k+i]
                dist = np.sqrt(np.sum( xaya ** 2))
                tmp = tmp+dist
        v_cm_s.append(tmp * (3.0 / 7.0)) #313 filter
    return  np.array(v_cm_s)

def cal_sta(dataset, num_par: int, neuron_id: int) -> NDArray:
    """Calculate spike triggered average.

    Parameters
    -----------
    dataset: 
        load dataset.
    num_par: int
        number of areas to discretize the open field.
    neuron_id: int
        the index for neurons.
    
    Return
    ----------
    sta : NDArray
        the spike triggered average of the selected neuron.
    """
    speed = cal_velocity(dataset.coords_xy)
    binned_position = bin_pos(dataset.coords_xy[speed > 1.0], num_par) # use the data whose speed is larger than 1cm/s
    filtered_spikes = dataset.spikes[speed > 1.0, neuron_id] 

    sta = np.zeros(num_par*num_par)
    for position in sorted(np.unique(binned_position)):
        index = (binned_position == position)
        spikes_sum = np.sum(filtered_spikes[index])
        total_time = np.sum(index)
        sta[int(position)-1] = spikes_sum/total_time
    sta = np.array(sta).reshape(num_par,-1)
    
    return sta

def get_3sigma(results_all: list, neuron_id: int) -> Tuple:
    """Get the 3 sigma from the shuffled MI.
    
    Return
    ----------
    behavior_3sigma : float
        3sigma for behavior shuffled MI.
    event_3sigma : float
        3sigma for event shuffled MI.
    """
    beh_std_3 = np.array(results_all['behavior shuffled MI all'])[:,neuron_id].std()*3
    event_std_3 = np.array(results_all['event shuffled MI all'])[:,neuron_id].std()*3

    beh_mu = np.array(results_all['behavior shuffled MI all'])[:,neuron_id].mean()
    event_mu = np.array(results_all['event shuffled MI all'])[:,neuron_id].mean()

    behavior_3sigma = beh_mu + beh_std_3
    event_3sigma = event_mu + event_std_3
    return behavior_3sigma, event_3sigma

def get_pc_ratio(results_all:list) -> Tuple:
    """Get place cells ratio based on two shuffle methods.

    Return
    ----------
    pc_beh_id : list
        place cell ratio from behavior shuffling method.
    pc_event_id : list
        place cell ratio from event shuffling method.
    """
    pc_beh_id, pc_event_id = [], []
    for neuron_id in range(len(results_all['original MI'])):
        behavior_3sigma, event_3sigma = get_3sigma(results_all, neuron_id)
        if results_all['original MI'][neuron_id] > behavior_3sigma:
            pc_beh_id.append(neuron_id)
        if results_all['original MI'][neuron_id] > event_3sigma:
            pc_event_id.append(neuron_id)
    return (pc_beh_id, pc_event_id)

def softmax(x: NDArray) -> NDArray:
    """Return the softmax of the input vector x.
    """
    out = np.exp(x - np.max(x)) # to prevent data overflow
    for i in range(len(x)):
        out[i] /= np.sum(out[i])
    return out

def downsample(X: NDArray, y: NDArray) -> Tuple:
    """Downsample X and y based on the minor class in y.
    
    Randomly select the samples in major classes in y and X accordingly.
    """
    classes, counts = np.unique(y, return_counts=True)
    classes_resample = classes[classes != classes[np.argmin(counts)]]
    count_min = np.min(counts)
    X_new = X[y==classes[np.argmin(counts)]]
    y_new = y[y==classes[np.argmin(counts)]]
    for c in classes_resample:
        X_tmp, y_tmp = resample(X[y==c], y[y==c], n_samples=count_min)
        X_new = np.vstack((X_new, X_tmp))
        y_new = np.append(y_new, y_tmp)
    return X_new, y_new

def segment(a: NDArray):
    """Segment array based on continuous positions.

    Return
    ------
    seg_ind: list
        an array of segmentation indices.
    """
    if len(a) == 1: return [0]
    if len(a) == 2: return [1] if a[0]==a[1] else [0]
    seg_ind = []
    for i in range(len(a)-1):
        if a[i] != a[i+1]:
            seg_ind.append(i+1)
    return seg_ind