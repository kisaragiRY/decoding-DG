import pytest
import numpy as np
from modules.func import *

def test_load_data():
    """Test load_data() in func.py module
    """
    all_data_dir=Path('data/alldata/') # data directory
    datalist=[x for x in all_data_dir.iterdir()] # get the list of files under the data directory

    for data_dir in datalist:
        assert load_data(data_dir) # load data