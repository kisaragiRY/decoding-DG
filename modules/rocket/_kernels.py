"""
Modification of the code authored by angus924.
https://github.com/sktime/sktime/blob/main/sktime/transformations/panel/rocket/_rocket_numba.py
"""

from typing import Tuple, Optional
from numpy.typing import NDArray

from dataclasses import dataclass

import numpy as np

def _generate_1d_kernels(num_features: int, num_timepoints: int, num_kernels: int, seed: Optional[int] = None) -> Tuple:
    """Generate random 1d kernels.

    Parameters
    ----------
    num_features: int
        the number of features.
    num_timepoints: int
        the number of time points of each instance.
    num_kernels: int
        the number of kernels to apply to the data.
    seed: Optional[int] = None
        for controling the random state.
    
    Return
    ----------
    random 1d kernels.
        weights: weights for each kernel;
        length_list: length of each kernel;
        biases: bias assigned to each kernel;
        dilations: dilation assigned to each kernel;
        paddings: padding assigned to each kernel;
        num_features_list: how many features to use for each kernel;
        features_indices: the corresponding feature indices for num_features_list.
    """
    if seed is not None:
        np.random.seed(seed)
    
    candidate_length_list = np.array((7, 9, 11), dtype=np.int32)
    length_list = np.random.choice(candidate_length_list, num_kernels).astype(np.int32)

    num_features_list = np.zeros(num_kernels, dtype=np.int32) # how many features to use per kernel
    for i in range(num_kernels):
        limit = min(num_features, length_list[i])
        num_features_list[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    features_indices = np.zeros(num_features_list.sum(), dtype=np.int32) 

    weights = np.zeros(
        np.int32(
            np.dot(length_list.astype(np.float32), num_features_list.astype(np.float32))
        ),
        dtype=np.float32,
    )
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    s_w = 0  # start index for weights
    s_f = 0  # start index for features

    for i in range(num_kernels):

        _length = length_list[i]
        _num_features = num_features_list[i]

        _weights = np.random.normal(0, 1, _num_features * _length).astype(
            np.float32
        )

        e_w = s_w + (_num_features * _length)
        e_f = s_f + _num_features

        s_w_per_k = 0  # start index of weight per kernel
        for _ in range(_num_features):
            e_w_per_k = s_w_per_k + _length
            _weights[s_w_per_k:e_w_per_k] = _weights[s_w_per_k:e_w_per_k] - _weights[s_w_per_k:e_w_per_k].mean()
            s_w_per_k = e_w_per_k

        weights[s_w:e_w] = _weights

        features_indices[s_f:e_f] = np.random.choice(
            np.arange(0, num_features), _num_features, replace=False
        )# which features to use per kernel (could overlap)

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(
            0, np.log2((num_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        s_w = e_w
        s_f = e_f

    return (
        weights,
        length_list,
        biases,
        dilations,
        paddings,
        num_features_list,
        features_indices,
    )

def _generate_nd_kernels(num_features: int, num_timepoints: int, num_kernels: int, kernel_dim: int,seed: Optional[int] = None) -> Tuple:
    """Generate random nd kernels.

    Parameters
    ----------
    num_features: int
        the number of features of the original data X.
    num_timepoints: int
        the number of time points of each instance from the original data X.
    num_kernels: int
        the number of kernels to apply to the data.
    seed: Optional[int] = None
        for controling the random state.
    
    Return
    ----------
    random nd kernels.
        weights: weights for each kernel;
        length_list: length of each kernel;
        biases: bias assigned to each kernel;
        dilations: dilation assigned to each kernel;
        paddings: padding assigned to each kernel;
        num_combinations_list: the number of combinations of features to use for 
                               each kernel;
        combinations_indices: the corresponding feature indices of one combination 
                               from num_features_list.
    """
    if seed is not None:
        np.random.seed(seed)
    
    candidate_length_list = np.array((7, 9, 11), dtype=np.int32)
    length_list = np.random.choice(candidate_length_list, num_kernels).astype(np.int32)

    num_combinations_list = np.zeros(num_kernels, dtype=np.int32) # how many features to use per kernel
    for i in range(num_kernels):
        limit = min(num_features, length_list[i])
        num_combinations_list[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    combinations_indices = np.zeros(num_combinations_list.sum(), dtype=np.int32) 

    weights = np.zeros(
        np.int32(
            np.dot(length_list.astype(np.float32), num_combinations_list.astype(np.float32))
        ) * kernel_dim,
        dtype=np.float32,
    )
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    features_combinations = np.array(
                    np.meshgrid((np.arange(0, num_features), np.arange(0, num_features)))
                    ).T.reshape(-1, kernel_dim)

    s_w = 0  # start index for weights
    s_f = 0  # start index for features

    for i in range(num_kernels):

        _length = length_list[i]
        _num_combinations = num_combinations_list[i] # select ramdom number of features

        _weights = np.random.normal(0, 1, _num_combinations * _length * kernel_dim).astype(
            np.float32
        )

        e_w = s_w + (_num_combinations * _length * kernel_dim)
        e_f = s_f + _num_combinations

        s_w_per_k = 0  # start index of weight per kernel
        for _ in range(_num_combinations):
            e_w_per_k = s_w_per_k + _length * kernel_dim
            _weights[s_w_per_k:e_w_per_k] = _weights[s_w_per_k:e_w_per_k] - _weights[s_w_per_k:e_w_per_k].mean()
            s_w_per_k = e_w_per_k

        weights[s_w:e_w] = _weights

        combinations_indices[s_f:e_f] = np.random.choice(
            features_combinations, _num_combinations, replace=False
        )# which features combinations to use per kernel (could overlap)

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(
            0, np.log2((num_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        s_w = e_w
        s_f = e_f

    return (
        weights,
        length_list,
        biases,
        dilations,
        paddings,
        num_combinations_list,
        combinations_indices,
    )

def _apply_1d_kernel(X_ins: NDArray, kernel: Tuple):  
    """Apply the kernel to the one instance of X
    """
    _, num_timepoints = X_ins.shape
    (
        weights, 
        kernel_length,
        bias,
        dilation,
        padding,
        num_features_list, # number of features to use
        features_indices, # indices of features to use
    ) = kernel # one kernel

    output_length = (num_timepoints + (2 * padding)) - ((kernel_length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (num_timepoints + padding) - ((kernel_length - 1) * dilation)

    for conv_i in range(-padding, end):

        _sum = bias

        current_time = conv_i

        for weight_i in range(kernel_length):

            if current_time > -1 and current_time < num_timepoints:

                for feature_i in range(num_features_list):
                    _sum += weights[feature_i, weight_i] * X_ins[features_indices[feature_i], current_time]

            current_time += dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return np.float32(_ppv / output_length), np.float32(_max)

def _apply_nd_kernel(X_ins: NDArray, kernel: Tuple, kernel_dim: int = 2):  
    """Apply the kernel to the one instance of X
    """
    _, num_timepoints = X_ins.shape
    (
        weights, 
        kernel_length,
        bias,
        dilation,
        padding,
        num_combinations, # number of combinations
        combinations_indices, # features's index within each combination
    ) = kernel # one kernel

    output_length = (num_timepoints + (2 * padding)) - ((kernel_length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (num_timepoints + padding) - ((kernel_length - 1) * dilation)

    for conv_i in range(-padding, end):

        _sum = bias

        current_time = conv_i

        for weight_i in range(kernel_length):
            if current_time > -1 and current_time < num_timepoints:
                for combination_i in range(num_combinations):
                    combinations = combinations_indices[combination_i]
                    for feature_i in range(len(combinations)):
                        _sum += weights[combination_i, weight_i, feature_i] * X_ins[combinations[feature_i], current_time]

            current_time += dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return np.float32(_ppv / output_length), np.float32(_max)

def _apply_kernels(X: NDArray, kernels: Tuple, kernel_dim: int):
    """Apply the kernels to X.
    """
    if kernel_dim == 1:
        (
            weights,
            length_list,
            biases,
            dilations,
            paddings,
            num_features_list,
            features_indices,
        ) = kernels
    else:
        (
            weights,
            length_list,
            biases,
            dilations,
            paddings,
            num_combinations_list,
            combinations_indices,
        ) = kernels

    n_instances, _, _ = X.shape
    num_kernels = len(length_list)

    _X = np.zeros(
        (n_instances, num_kernels * 2), dtype=np.float32
    )  # 2 features per kernel

    for i in range(n_instances):

        s_w = 0  # start index of weights
        s_f = 0  # start index of features from the orginal data to use
        s_out_f = 0  # start index of the features for the output

        for j in range(num_kernels):

            e_w = s_w + num_features_list[j] * length_list[j] * kernel_dim # end index of weights
            e_f = s_f + num_features_list[j] # end index of features to use
            e_out_f = s_out_f + 2 # end index of the features for the output

            if kernel_dim == 1:
                _weights = weights[s_w:e_w].reshape((num_features_list[j], length_list[j]))

                kernel = (
                        _weights,
                        length_list[j],
                        biases[j],
                        dilations[j],
                        paddings[j],
                        num_features_list[j],
                        features_indices[s_f:e_f]
                        )
            
                _X[i, s_out_f:e_out_f] = _apply_1d_kernel(
                    X[i],
                    kernel
                )
            else:
                _weights = weights[s_w:e_w].reshape((num_combinations_list[j], length_list[j], kernel_dim))

                kernel = (
                        _weights,
                        length_list[j],
                        biases[j],
                        dilations[j],
                        paddings[j],
                        num_combinations_list[j],
                        combinations_indices[s_f:e_f]
                        )
                _X[i, s_out_f:e_out_f] = _apply_nd_kernel(
                    X[i],
                    kernel,
                    kernel_dim
                )

            s_w = e_w
            s_f = e_f
            s_out_f = e_out_f

    return _X.astype(np.float32)
