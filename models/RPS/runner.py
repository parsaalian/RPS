import numpy as np
import networkx as nx
from node2vec import Node2Vec
from pprint import pprint

from data import *
from utils.weight_transformers import *


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