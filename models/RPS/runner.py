import numpy as np
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm

from data import *
from utils.weight_transformers import *
from utils.clustering_methods import *
from utils.weight_functions import *
from utils.financial_measures import calculate_measures
from utils.measures import *


def create_distance_graph(history_df, transformer):
    graph = nx.Graph()
    stock_names = history_df.columns
    dist_mat = history_df.corr().fillna(1)
    
    for stock1 in stock_names:
        for stock2 in stock_names:
            weight = transformer(dist_mat[stock1][stock2])
            graph.add_weighted_edges_from([(stock1, stock2, weight), (stock2, stock1, weight)])
    return graph


def df_list_to_readable(df, columns):
    copy_df = df.copy()
    for column in columns:
        copy_df[column] = df[column].apply(
            lambda x: '//'.join(list(map(str, x))))
    return copy_df


def readable_to_df_list(df, columns):
    def transform_back(x):
        try:
            return list(map(eval, x.split('//')))
        except:
            try:
                return x.split('//')
            except:
                return x

    copy_df = df.copy()
    for column in columns:
        copy_df[column] = df[column].apply(transform_back)
    return copy_df


class RPSRunner:
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
        
        distance_graph = create_distance_graph(
            train_dataset,
            eval(self.model_config.distance_transformer)
        )
        assets = list(distance_graph.nodes())
        
        if 'embedding_path' in self.train_config:
            vectors = np.load(self.train_config.embedding_path)
        else:
            model = Node2Vec(
                distance_graph,
                dimensions=self.model_config.dimensions,
                walk_length=self.model_config.walk_length,
                num_walks=self.model_config.num_walks,
                workers=self.model_config.workers,
            ).fit(
                window=self.model_config.window,
                min_count=self.model_config.min_count,
                batch_words=self.model_config.batch_words,
            )
            
            vectors = np.array([model.wv[node] for i, node in enumerate(distance_graph.nodes())])
            
            np.save('{0}/embeddings.npy'.format(self.save_dir), vectors)
        
        baskets = eval(self.train_config.clustering_method)(vectors, assets, self.train_config)
        
        results = []
        
        for i in tqdm(range(len(baskets))):
            try:
                assets = baskets[i]
                weight_dict = dict(eval(self.train_config.weight_method)(train_dataset[assets]))
                # print(weight_dict)
                assets, weights = list(weight_dict.keys()), list(weight_dict.values())
                results.append([
                    assets,
                    weights,
                    *calculate_measures(assets, train_dataset, weights)
                ])
            except Exception as e:
                print(e)
        
        results = np.asarray(results, dtype=object)
                
        df = pd.DataFrame({
            self.output_columns[i]: results[:, i] for i in range(len(self.output_columns))
        })
        df = df_list_to_readable(df, ['stocks', 'weights'])
        df.to_csv(self.save_dir + '/results.csv')
    
    
    def test(self):
        test_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config)
        
        weights_df = readable_to_df_list(pd.read_csv(self.test_config.train_results), ['stocks', 'weights'])
        
        # future performance measure
        
        future_performances = []
        
        for _, row in weights_df.iterrows():
            future_performances.append([
                row.stocks,
                row.weights,
                *calculate_measures(row.stocks, test_dataset, row.weights)
            ])
        
        future_performances = np.asarray(future_performances, dtype=object)
                
        df = pd.DataFrame({
            self.output_columns[i]: future_performances[:, i] for i in range(len(self.output_columns))
        })
        df = df_list_to_readable(df, ['stocks', 'weights'])
        df.to_csv(self.save_dir + '/future_performances.csv')