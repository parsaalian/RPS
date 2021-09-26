import numpy as np
import pandas as pd


def indextrack_loader(dataset_config, run_config):
    with open(dataset_config.data_path, 'r') as f:
        data = f.read()
        data = list(map(float, data.split()))
        asset_count = int(data[0])
        prices = data[2:]
        history = pd.DataFrame(np.array(np.split(np.array(prices), asset_count + 1)).swapaxes(0, 1))
        return history