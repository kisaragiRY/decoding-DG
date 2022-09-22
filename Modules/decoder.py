import numpy as np

class linear_gaussian():
    '''
    a linear guassian model
    x_t=theta・n_t + b_t
    x_t: discretized position
    theta: parameter
    n_t: spikes
    b_t: intercept
    '''
    def __init__(self,position:np.array,spikes:np.array) -> None:
        self.position=position
        self.spikes=position