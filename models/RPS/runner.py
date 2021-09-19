import numpy as np
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm

from data import *
from utils.weight_transformers import *
from utils.clustering_methods import *
from utils.financial_measures import calculate_measures


def create_distance_graph(history_df, transformer):
    graph = nx.Graph()
    stock_names = history_df.columns
    dist_mat = history_df.corr().fillna(1)
    
    for stock1 in stock_names:
        for stock2 in stock_names:
            weight = transformer(dist_mat[stock1][stock2])
            graph.add_weighted_edges_from([(stock1, stock2, weight), (stock2, stock1, weight)])
    return graph


class RPSRunner:
    def __init__(self, config):
        self.save_dir = config.save_dir
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.train_config = config.train
        self.test_config = config.test
    
    
    def train(self):
        train_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.train_config)
        
        distance_graph = create_distance_graph(
            train_dataset,
            eval(self.model_config.distance_transformer)
        )
        
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
        
        baskets = eval(self.train_config.clustering_method)(self.train_config)
        
        results = []
        
        for i in tqdm(range(len(baskets))):
            assets = baskets[i]
            try:
                weights = eval(self.train_config.weight_method)(train_dataset[assets])
                results.append(
                    assets,
                    weights,
                    *calculate_measures(assets, train_dataset, weights)
                )
            except Exception as e:
                print(e)