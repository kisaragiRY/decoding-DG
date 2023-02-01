# dataset.py
## `BaseDataset`
A base dataset that requires three inputs: 
1. `data_dir`: the data directory to the raw data.
2. `mobility`: whether to filter data based on mobility, if does, specify a float number.
3. `shuffle_method`: whether to shuffle the data, if does, specify the shuffling method.


it has 3 methods.
1. `_load_data()`: load the raw csv data.
2. `_shuffle()`: shuffle the raw data.
3. `split_data()`: a public method that split the raw data into train set and test set.

Upon initializing, 
1. load the raw data
2. determine whether filtering mobility data is needed
3. determine whether shuffling is needed
4. save the raw data (shuffled or not) into X and y.


## `SmoothedSpikesDataset(BaseDataset)`
A data which uses a gaussian kernel to smooth the spikes.

it extends 2 methods on the basis of `BaseDataset`:
1. `_filter_spikes()`: a private method to smooth spikes using Gaussian kernel.
2. `load_all_data()`: a public method to load smoothed train and test data.