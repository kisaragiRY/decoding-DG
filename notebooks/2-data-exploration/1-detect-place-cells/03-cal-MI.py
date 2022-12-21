import pickle
from tqdm import tqdm

from metrics import InfoMetrics
from dataloader import BaseDataset
from param import *
from util import bin_pos

def cal_MI():
    """Calculate MI.

    based on original data and shuffled data.
    """
    for data_dir in tqdm(ParamDir().data_list):
        data_name = str(data_dir).split('/')[-1]

        beh_dataset = BaseDataset(data_dir, 'behavior shuffling')
        event_dataset = BaseDataset(data_dir, 'events shuffling')

        binned_position = bin_pos(beh_dataset.coords_xy, 2) # discretized into 2x2 grid
        shuffled_position = bin_pos(beh_dataset.shuffled_coords_xy, 2)

        shuffled_spikes = event_dataset.shuffle_spikes

        info = InfoMetrics(beh_dataset.spikes, binned_position)
        beh_shuffled_info = InfoMetrics(beh_dataset.spikes, shuffled_position)
        event_shuffled_info = InfoMetrics(shuffled_spikes, binned_position)

        original_MI, beh_shuffled_MI, event_shuffled_MI = [], [], []
        for n_id in range(beh_dataset.spikes.shape[1]):
            original_MI.append(info.mutual_info(n_id)) # orignal MI
            beh_shuffled_MI.append(beh_shuffled_info.mutual_info(n_id)) # behavior shuffled MI
            event_shuffled_MI.append(event_shuffled_info.mutual_info(n_id)) # event shuffled MI
        
        result_MI = {
            "original MI": original_MI,
            "behavior shuffled MI": beh_shuffled_MI,
            'event shuffled MI': event_shuffled_MI
        }
        if not (ParamDir().output_dir/data_name).exists():
            (ParamDir().output_dir/data_name).mkdir()
        with open(ParamDir().output_dir/data_name/"MI_all.pickle", "wb") as f:
            pickle.dump(result_MI, f)


if __name__ == "__main__":
    cal_MI()