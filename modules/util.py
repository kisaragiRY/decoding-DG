from numpy.typing import NDArray
from typing import Tuple

import numpy as np
from numpy.linalg import det, inv


def gauss1d(xx: NDArray, mu: float = 0, sigma: float = .2):
    """A Gaussian kernel."""
    kernel = 1 / ((2 * np.pi) ** 2 * sigma) * np.exp(- (xx - mu) ** 2 / (2 * sigma ** 2))
    if sum(kernel) == 0: # if xx is array of zeros then return itself
        return xx
    else: 
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

def estimate_firing_rate(spikes: NDArray, window_size: int, sigma: float = .2) -> NDArray:
    """Smooth the spikes with a gaussian kernel.
    
    Parameter
    ------------
    window_size: int
        the size of the gaussian kernel
    sigma: float
        the sigma of the gaussian kernel
    spikes: NDArray
        the spikes count data. 
    """
    kernel = gauss1d(np.linspace(-3, 3, window_size), sigma= sigma)

    def filtered(x: NDArray) -> NDArray:
        """Convovle with the given kernel."""
        return np.convolve(x, kernel, mode="same")

    return np.apply_along_axis(filtered, 0, spikes)

def spilt_data(X: np.array, y: np.array, train_ratio: float) -> Tuple[Tuple[np.array,np.array],Tuple[np.array,np.array]]:
    """Get training and testing data."""
    train_size = int( len(X) * train_ratio )
    X_train,y_train = X[:train_size], y[:train_size]
    X_test,y_test = X[train_size:], y[train_size:]
    return (X_train, y_train), (X_test, y_test)