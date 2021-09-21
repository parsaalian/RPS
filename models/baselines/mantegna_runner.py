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
        try:
            assets = baskets[i]
            weight_dict = dict(eval(model_config.weight_method)(history_df[assets]))
            # print(weight_dict)
            assets, weights = list(weight_dict.keys()), list(weight_dict.values())
            results.append([
                assets,
                weights,
                *calculate_measures(assets, history_df, weights)
            ])
        except Exception as e:
            print(e)
    
    results = np.asarray(results, dtype=object)
    
    print(results)
    
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
        test_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config)
        
        weights_df = readable_to_df_list(pd.read_csv(self.test_config.train_results), ['stocks', 'weights'])
        
        df_future_performance(
            test_dataset,
            weights_df,
            self.output_columns,
            self.save_dir + '/future_performances.csv'
        )