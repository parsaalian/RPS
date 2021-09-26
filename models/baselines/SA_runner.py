import numpy as np
import pandas as pd
from pypfopt.expected_returns import ema_historical_return
from tqdm import tqdm

from data import *
from utils.financial_measures import *
from utils.measures import *


def train_and_save_sa_model(history_df, C, model_config):
    mu = ema_historical_return(history_df)
    
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


def train_multi_sa_models(
    history_df,
    C,
    model_config,
    train_config,
    save_dir,
    save_path='sa_weights'
):
    output_columns = [
        'stocks', 'weights',
        'corr_min', 'corr_max', 'corr_mean', 'corr_std',
        'return', 'sigma', 'sharpe', 'information', 'modigliani',
    ]
    
    results = []
            
    for _ in tqdm(range(train_config.count)):
        measures = train_and_save_sa_model(history_df, C, model_config)
        weights = measures[0][np.where(measures[0] > 0)]
        assets = history_df.columns[np.where(measures[0] > 0)]
        results.append([assets, weights, *measures[1:]])
    
    results = np.asarray(results, dtype=object)
    
    df = pd.DataFrame({
        output_columns[i]: results[:, i] for i in range(len(output_columns))
    })
    df = df_list_to_readable(df, ['stocks', 'weights'])
    df.to_csv('{0}/{1}.csv'.format(save_dir, save_path), index=False)
    df = readable_to_df_list(df, ['stocks', 'weights'])
    
    return df


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
        train_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.train_config)
        
        df = train_multi_sa_models(
            train_dataset,
            train_dataset.cov(),
            self.model_config,
            self.train_config,
            self.save_dir
        )
        
        if self.test_config.test_method == 'future_performance':
            test_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config)
    
            df_future_performance(test_dataset, df, self.output_columns, self.save_dir + '/future_performance.csv')
        elif self.test_config.test_method == 'noise_stability':
            train_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.train_config)
            
            cov_to_corr = train_dataset.cov().divide(train_dataset.corr())
            corrs = train_dataset.corr().fillna(1)
            noise = np.random.normal(0, self.test_config.noise_sigma, corrs.shape)
            noised = corrs + noise
            noised = noised.clip(lower=-1, upper=1)
            np.fill_diagonal(noised.values, 1)
            noised.to_csv(self.save_dir + '/noised_correlations.csv', index=False)
            C = noised.multiply(cov_to_corr)
            
            new_df = train_multi_sa_models(
                train_dataset,
                C,
                self.model_config,
                self.train_config,
                self.save_dir
            )
            
            new_df = new_df.sort_values(self.test_config.sort_column).reset_index(drop=True)
            
            df = df.sort_values(self.test_config.sort_column).reset_index(drop=True)
            
            stability_df = pd.DataFrame(index=list(range(len(new_df))), columns=list(range(len(df))))
            
            for i in range(len(new_df)):
                for j in range(len(df)):
                    distance = calculate_noise_stability(
                        set(new_df.loc[i, 'stocks']),
                        set(df.loc[j, 'stocks'])
                    )
                    stability_df.loc[i, j] = distance
            
            stability_df.to_csv(self.save_dir + '/stability_matrix.csv', index=False)
        elif self.test_config.test_method == 'time_stability':
            train_dataset1 = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config.test1)
            train_dataset2 = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config.test2)
            
            df1 = train_multi_sa_models(
                train_dataset1,
                train_dataset1.cov(),
                self.model_config,
                self.train_config,
                self.save_dir,
                'sa_weights1'
            )
            
            df2 = train_multi_sa_models(
                train_dataset2,
                train_dataset2.cov(),
                self.model_config,
                self.train_config,
                self.save_dir,
                'sa_weights2'
            )
            
            df1 = df1.sort_values(self.test_config.sort_column).reset_index(drop=True)
            df2 = df2.sort_values(self.test_config.sort_column).reset_index(drop=True)
            
            stability_df = pd.DataFrame(index=list(range(len(df1))), columns=list(range(len(df2))))
            
            for i in range(len(df1)):
                for j in range(len(df2)):
                    distance = calculate_noise_stability(
                        set(df1.loc[i, 'stocks']),
                        set(df2.loc[j, 'stocks'])
                    )
                    stability_df.loc[i, j] = distance
            stability_df.to_csv(self.save_dir + '/stability_matrix.csv', index=False)
        