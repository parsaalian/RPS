import numpy as np
import pandas as pd
from pypfopt.expected_returns import ema_historical_return
from tqdm import tqdm

from data import *
from utils.financial_measures import *
from utils.measures import *


def train_and_save_sa_model(history_df, model_config):
    mu = ema_historical_return(history_df)
    C = history_df.cov()
    
    N = len(history_df.columns)
    K = model_config.cardinality
    
    randoms = np.random.uniform(0, 1, (K))
    randoms = list(randoms / randoms.sum())
    w = np.random.permutation([*randoms, *[0 for i in range(N - K)]])
    ws = w.copy()

    gamma = model_config.gamma
    gamma_u = model_config.gamma_u
    u = model_config.u
    T = model_config.T
    
    L = lambda w, mu, C : gamma * np.matmul(mu, w.transpose()) - (1 - gamma) * np.matmul(w, np.matmul(C, w.transpose()))
    
    for _ in range(2 * N):
        wu = w.copy()
        choices = np.where(w != 0.0)[0]
        i = np.random.choice(choices)
        j = np.random.randint(N)
        if w[j] == 0:
            wu[j] = w[i]
            wu[i] = 0
        else:
            ui = ru = np.random.uniform(0, min(u, w[i], w[j]))
            wu[j] = w[j] + ui
            wu[i] = w[i] - ui
        dl = L(wu, mu, C) - L(w, mu, C)
        if dl > 0:
            w = wu.copy()
            if L(w, mu, C) > L(ws, mu, C):
                ws = w.copy()
        else:
            p = np.random.uniform(0, 1)
            if p <= np.exp(dl / T):
                w = wu.copy()
        T = gamma * T
        u = gamma_u * u
    
    return calculate_measures(history_df.columns, history_df, ws)


class SARunner:
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
    
    
    def test(self):
        results = []
        
        for _ in tqdm(range(self.train_config.count)):
            train_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.train_config)
            measures = train_and_save_sa_model(train_dataset, self.model_config)
            weights = measures[0][np.where(measures[0] > 0)]
            assets = train_dataset.columns[np.where(measures[0] > 0)]
            results.append([assets, weights, *measures[1:]])
        
        results = np.asarray(results, dtype=object)
        
        df = pd.DataFrame({
            self.output_columns[i]: results[:, i] for i in range(len(self.output_columns))
        })
        df = df_list_to_readable(df, ['stocks', 'weights'])
        df.to_csv('{0}/{1}.csv'.format(self.save_dir, 'sa_weights'), index=False)
        
        df = readable_to_df_list(df, ['stocks', 'weights'])
        
        test_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config)
        
        df_future_performance(test_dataset, df, self.output_columns, self.save_dir + '/future_performance.csv')