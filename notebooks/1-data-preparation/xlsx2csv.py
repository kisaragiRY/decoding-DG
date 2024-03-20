from pathlib import Path
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = Path('data/raw/')
output_dir = Path('data/processed/')
datalist = np.array([x for x in DATA_DIR.iterdir()])
for data_dir in tqdm(datalist):
    data_name = str(data_dir).split('/')[-1]
    if not (output_dir/data_name).is_dir():
        (output_dir/data_name).mkdir()
    coords_df = pd.read_excel(data_dir/'position.xlsx')
    coords_df.to_csv(output_dir/data_name/'position.csv')

    spikes_df = pd.read_excel(data_dir/'traces.xlsx',index_col=0)
    spikes_df.to_csv(output_dir/data_name/'traces.csv')