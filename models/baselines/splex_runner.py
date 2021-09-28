import numpy as np
import networkx as nx
from tqdm import tqdm

from data import *
from utils.readable_df import *
from utils.weight_functions import *
from utils.financial_measures import calculate_measures
from utils.measures import *


class SplexSolver:
    def __init__(self, G, s=2):
        self.G = G
        self.s = s
        self.N = len(G.nodes())
        self.nodes_list = list(G.nodes())
        self.nodes_index = {x: i for (i, x) in enumerate(self.nodes_list)}
        self.max = 0
        self.maxi = [0 for i in range(self.N)]
        self.res = []
        self.order = []

    def w(self, X):
        res = 0
        for i in X:
            res += self.G.nodes[i]["weight"]
        return res

    def is_splex(self, X):
        Gp = self.G.subgraph(X)
        for x in Gp.nodes():
            if Gp.degree(x) < len(Gp.nodes()) - self.s:
                return False
        return True

    def min_order(self, X):
        mini, m = self.N, self.N
        for v in X:
            if self.order.index(self.nodes_index[v]) < mini:
                mini = self.order.index(self.nodes_index[v])
                m = v
        return mini, m

    def find_max(self, C, P):
        if len(C) == 0:
            if self.w(P) > self.max:
                self.max = self.w(P)
                self.res = P
        while len(C) != 0:
            if self.w(C) + self.w(P) <= self.max:
                return
            ic, i = self.min_order(C)
            if self.maxi[ic] + self.w(P) <= self.max:
                return
            C = C - {i}
            Pp = {i}.union(P)
            Cp = set()
            for v in C:
                if self.is_splex(Pp.union({v})):
                    Cp.add(v)
            self.find_max(Cp, Pp)

    def solve(self):
        self.order = list(np.argsort([self.G.nodes[i]["weight"] for i in self.nodes_list]))
        for i in tqdm(list(reversed(range(self.N)))):
            C = set([self.nodes_list[self.order[x]] for x in range(i + 1, self.N)])
            self.find_max(C, {self.nodes_list[self.order[i]]})
            self.maxi[i] = self.max


def train_and_save_splex(
    history_df,
    correlation_matrix,
    model_config,
    save_dir
):
    output_columns = [
        'stocks', 'weights',
        'corr_min', 'corr_max', 'corr_mean', 'corr_std',
        'return', 'sigma', 'sharpe', 'information', 'modigliani',
    ]
    
    asset_performances = history_df.tail(1).reset_index(drop=True) / history_df.head(1).reset_index(drop=True) - 1
    
    G = nx.Graph()
        
    for asset in asset_performances.columns:
        G.add_node(asset, weight=asset_performances.loc[0, asset])
    
    for i in correlation_matrix.index:
        for j in correlation_matrix.columns:
            i_j_corr = correlation_matrix.loc[i, j]
            if i_j_corr < model_config.weight_limit:
                G.add_edge(i, j, weight=i_j_corr)
    
    solver = SplexSolver(G, model_config.s)
    solver.solve()
    
    baskets = [list(solver.res)]
    
    results = []
    
    for i in tqdm(range(len(baskets))):
        try:
            assets = baskets[i]
            weight_dict = dict(eval(model_config.weight_method)(history_df[assets], model_config))
            # print(weight_dict)1
            assets, weights = list(weight_dict.keys()), list(weight_dict.values())
            results.append([
                assets,
                *calculate_measures(assets, history_df, weights)
            ])
        except Exception as e:
            print(e)
    
    results = np.asarray(results, dtype=object)
    df = pd.DataFrame({
        output_columns[i]: results[:, i] for i in range(len(output_columns))
    })
    df = df_list_to_readable(df, ['stocks', 'weights'])
    df.to_csv('{0}/{1}.csv'.format(save_dir, 'splex_results'), index=False)
    
    return df


class SplexRunner:
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
        
        train_and_save_splex(
            train_dataset,
            train_dataset.corr().fillna(1),
            self.model_config,
            self.save_dir,
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
        elif self.test_config.test_method == 'noise_stability':
            train_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.train_config)
            
            corrs = train_dataset.corr().fillna(1)
            noise = np.random.normal(0, self.test_config.noise_sigma, corrs.shape)
            noised = corrs + noise
            noised = noised.clip(lower=-1, upper=1)
            np.fill_diagonal(noised.values, 1)
            noised.to_csv(self.save_dir + '/noised_correlations.csv', index=False)
            
            df = train_and_save_splex(
                train_dataset,
                noised,
                self.model_config,
                self.save_dir,
            )
            
            df = df.sort_values(self.test_config.sort_column).reset_index(drop=True)
            
            pre_df = readable_to_df_list(pd.read_csv(self.test_config.train_results), ['stocks', 'weights'])
            pre_df = pre_df.sort_values(self.test_config.sort_column).reset_index(drop=True)
            
            stability_df = pd.DataFrame(index=list(range(len(df))), columns=list(range(len(pre_df))))
            
            for i in range(len(df)):
                for j in range(len(pre_df)):
                    distance = calculate_noise_stability(
                        set(df.loc[i, 'stocks']),
                        set(pre_df.loc[j, 'stocks'])
                    )
                    stability_df.loc[i, j] = distance
            
            stability_df.to_csv(self.save_dir + '/stability_matrix.csv', index=False)
        else:
            train_dataset1 = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config.test1)            
            train_dataset2 = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config.test2)
            
            df1 = train_and_save_splex(
                train_dataset1,
                train_dataset1.corr().fillna(1),
                self.model_config,
                self.save_dir,
            )
            
            df2 = train_and_save_splex(
                train_dataset2,
                train_dataset2.corr().fillna(1),
                self.model_config,
                self.save_dir,
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