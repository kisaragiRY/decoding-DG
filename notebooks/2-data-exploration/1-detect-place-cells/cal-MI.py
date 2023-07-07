import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import prange, njit

from metrics import InfoMetrics
from dataloader import BaseDataset
from param import *
from utils.util import bin_pos

def cal_all_MI(data_dir: Path):
    """Calculate MI.

    based on original data and shuffled data.
    """
    data_name = str(data_dir).split('/')[-1]

    base_dataset = BaseDataset(data_dir, ParamData().mobility, False, False)
    base_position = bin_pos(base_dataset.coords_xy, 2) # discretized into 2x2 grid
    base_spikes = base_dataset.spikes
    num_neurons = base_spikes.shape[1]

    original_MI = np.zeros(num_neurons)
    beh_shuffled_MI_all = np.zeros(ParamShuffle().num_repeat * num_neurons)
    event_shuffled_MI_all = np.zeros(ParamShuffle().num_repeat * num_neurons)

    for seed in tqdm(range(ParamShuffle().num_repeat)):
        beh_dataset = BaseDataset(data_dir, ParamData().mobility,'behavior shuffling', seed)
        event_dataset = BaseDataset(data_dir, ParamData().mobility, 'events shuffling', seed)

        shuffled_position = bin_pos(beh_dataset.coords_xy, 2)
        shuffled_spikes = event_dataset.spikes

        info = InfoMetrics(base_spikes, base_position)
        beh_shuffled_info = InfoMetrics(base_spikes, shuffled_position)
        event_shuffled_info = InfoMetrics(shuffled_spikes, base_position)

        for n_id in range(num_neurons):
            if seed == 0:
                original_MI[n_id] = info.cal_mi(n_id) # orignal MI
            beh_shuffled_MI_all[seed*num_neurons+n_id] = beh_shuffled_info.cal_mi(n_id) # behavior shuffled MI
            event_shuffled_MI_all[seed*num_neurons+n_id]  = event_shuffled_info.cal_mi(n_id) # event shuffled MI
        
    result_MI = {
        "original MI": original_MI,
        "behavior shuffled MI all": beh_shuffled_MI_all.reshape(ParamShuffle().num_repeat, num_neurons),
        'event shuffled MI all': event_shuffled_MI_all.reshape(ParamShuffle().num_repeat, num_neurons)
    }
    if not (ParamDir().output_dir/data_name).exists():
        (ParamDir().output_dir/data_name).mkdir()
    with open(ParamDir().output_dir/data_name/"MI_all.pickle", "wb") as f:
        pickle.dump(result_MI, f)


if __name__ == "__main__":

    Parallel(n_jobs=12)(delayed(
        cal_all_MI(data_dir)
        )(data_dir) for data_dir in ParamDir().data_list)