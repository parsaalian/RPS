import numpy as np
import pandas as pd
from tqdm import tqdm

from data import *
from utils.readable_df import *
from utils.weight_functions import *
from utils.financial_measures import calculate_measures
from utils.measures import *


def train_and_save_random_model(
    history_df,
    model_config,
    save_dir,
    save_path='random_results',
):
    output_columns = [
        'stocks', 'weights',
        'corr_min', 'corr_max', 'corr_mean', 'corr_std',
        'return', 'sigma', 'sharpe', 'information', 'modigliani',
    ]
    
    results = []
    
    port_counts = list(map(
        lambda x: abs(int(x)),
        np.random.normal(
            model_config.port_mean, model_config.port_std, model_config.count
        )
    ))
    for count in tqdm(port_counts):
        assets = list(np.random.choice(list(set(history_df.columns)), count, replace=False))
        if len(assets) > 1:
            try:
                weight_dict = dict(eval(model_config.weight_method)(history_df[assets], model_config))
                assets, weights = list(weight_dict.keys()), list(weight_dict.values())
                try:
                    results.append([
                        assets,
                        *calculate_measures(assets, history_df, weights),
                    ])
                except:
                    pass
            except:
                pass

    results = np.asarray(results)
    rand_df = pd.DataFrame({
        output_columns[i]: results[:, i] for i in range(len(output_columns))
    })
    
    rand_df = df_list_to_readable(rand_df, ['stocks', 'weights'])
    rand_df.to_csv('{0}/{1}.csv'.format(save_dir, save_path), index=False)
    
    return readable_to_df_list(rand_df, ['stocks', 'weights'])

class RandomRunner:
    def __init__(self, config):
        self.save_dir = config.save_dir
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.train_config = config.train
        self.test_config = config.test
        self.output_columns = [
            'stocks', 'weights',
            'corr_min', 'corr_max', 'corr_mean', 'corr_std',
            'return', 'sigma', 'sharpe', 'information', 'modigliani',
        ]
    
    
    def train(self):
        train_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.train_config)
        
        train_and_save_random_model(
            train_dataset, self.model_config, self.save_dir, save_path="random_results"
        )
    
    
    def test(self):
        if self.test_config.test_method == 'future_performance':
            test_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config)
            
            weights_df = readable_to_df_list(pd.read_csv(self.test_config.train_results), ['stocks', 'weights'])
            
            df_future_performance(
                test_dataset,
                weights_df,
                self.output_columns,
                self.save_dir + '/future_performances.csv'
            )
        else:
            raise Exception('Test method not supported for random')