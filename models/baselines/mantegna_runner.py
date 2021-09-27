import numpy as np
import networkx as nx
from networkx.algorithms.tree.mst import minimum_spanning_tree
from networkx.convert_matrix import to_scipy_sparse_matrix
from sknetwork.hierarchy import LouvainHierarchy
from sknetwork.hierarchy import cut_straight
from tqdm import tqdm

from data import *
from utils.readable_df import *
from utils.weight_functions import *
from utils.financial_measures import calculate_measures
from utils.measures import *


def train_and_save_mantegna_model(
    history_df,
    distance_matrix,
    model_config,
    save_dir,
    save_path='mantegna_results'
):
    output_columns = [
        'stocks', 'weights',
        'corr_min', 'corr_max', 'corr_mean', 'corr_std',
        'return', 'sigma', 'sharpe', 'information', 'modigliani',
    ]
    
    G = nx.Graph()

    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            G.add_edge(i, j, weight=distance_matrix.loc[i, j])
    
    T = minimum_spanning_tree(G)
    assets = list(T.nodes)
    
    sparse = to_scipy_sparse_matrix(T)
    
    louvain_hierarchy = LouvainHierarchy()
    dendrogram = louvain_hierarchy.fit_transform(sparse)
    
    labels = cut_straight(dendrogram)

    baskets = { i: [] for i in range(max(labels) + 1) }
    
    for i in range(len(labels)):
        baskets[labels[i]].append(assets[i])
    
    results = []
    
    for i in tqdm(range(len(baskets))):
        assets = baskets[i]
        if len(assets) > 1:
            try:
                weight_dict = dict(eval(model_config.weight_method)(history_df[assets], model_config))
                # print(weight_dict)
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
    df.to_csv('{0}/{1}.csv'.format(save_dir, save_path), index=False)
    
    return df


class MantegnaRunner:
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
        
        corrs = train_dataset.corr().fillna(1)
        d = corrs.apply(lambda x: np.sqrt(2 * (1 - x)))
        
        train_and_save_mantegna_model(
            train_dataset,
            d,
            self.model_config,
            self.save_dir,
            save_path='mantegna_results'
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
            d = noised.apply(lambda x: np.sqrt(2 * (1 - x)))
            
            df = train_and_save_mantegna_model(
                train_dataset,
                d,
                self.model_config,
                self.save_dir,
                save_path='mantegna_noised_results'
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
        elif self.test_config.test_method == 'time_stability':
            train_dataset1 = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config.test1)
            corrs1 = train_dataset1.corr().fillna(1)
            d1 = corrs1.apply(lambda x: np.sqrt(2 * (1 - x)))
            
            train_dataset2 = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config.test2)
            corrs2 = train_dataset1.corr().fillna(1)
            d2 = corrs2.apply(lambda x: np.sqrt(2 * (1 - x)))
            
            df1 = train_and_save_mantegna_model(
                train_dataset1,
                d1,
                self.model_config,
                self.save_dir,
                save_path='mantegna_results1'
            )
            
            df2 = train_and_save_mantegna_model(
                train_dataset2,
                d2,
                self.model_config,
                self.save_dir,
                save_path='mantegna_results2'
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