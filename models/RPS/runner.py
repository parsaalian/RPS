import numpy as np
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm

from data import *
from utils.readable_df import *
from utils.weight_transformers import *
from utils.clustering_methods import *
from utils.weight_functions import *
from utils.financial_measures import calculate_measures
from utils.measures import *


def create_distance_graph(assets, corrs, transformer):
    graph = nx.Graph()
    
    for stock1 in assets:
        for stock2 in assets:
            weight = transformer(corrs[stock1][stock2])
            graph.add_weighted_edges_from([(stock1, stock2, weight), (stock2, stock1, weight)])
    return graph


def train_and_save_node2vec_model(
    history_df,
    correlation_matrix,
    model_config,
    save_dir,
    embedding_path=None,
    result_path=None,
    save_paths={
        'results': 'results',
        'embeddings': 'embeddings'
    }
):
    output_columns = [
        'stocks', 'weights',
        'corr_min', 'corr_max', 'corr_mean', 'corr_std',
        'return', 'sigma', 'sharpe', 'information', 'modigliani',
    ]
    
    distance_graph = create_distance_graph(
        history_df.columns,
        correlation_matrix,
        eval(model_config.distance_transformer)
    )
    assets = list(distance_graph.nodes())
    
    if result_path is not None:
        pass
    elif embedding_path is not None:
        vectors = np.load(embedding_path)
    else:
        model = Node2Vec(
            distance_graph,
            dimensions=model_config.dimensions,
            walk_length=model_config.walk_length,
            num_walks=model_config.num_walks,
            workers=model_config.workers,
        ).fit(
            window=model_config.window,
            min_count=model_config.min_count,
            batch_words=model_config.batch_words,
        )
        
        vectors = np.array([model.wv[node] for i, node in enumerate(distance_graph.nodes())])
        
        np.save('{0}/{1}.npy'.format(save_dir, save_paths['embeddings']), vectors)
    
    if result_path is not None:
        df = readable_to_df_list(pd.read_csv(result_path), columns=['stocks', 'weights'])
    else:
        baskets = eval(model_config.clustering_method)(vectors, assets, model_config)
        
        results = []
        
        for i in tqdm(range(len(baskets))):
            assets = baskets[i]
            if assets is not None and len(assets) > 1:
                try:
                    weight_dict = dict(eval(model_config.weight_method)(history_df[assets], model_config))
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
        df.to_csv('{0}/{1}.csv'.format(save_dir, save_paths['results']), index=False)
    
    return readable_to_df_list(df, ['stocks', 'weights'])

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
        
        train_and_save_node2vec_model(
            train_dataset,
            train_dataset.corr().fillna(1),
            self.model_config,
            self.save_dir,
            self.train_config.embedding_path if 'embedding_path' in self.train_config else None.
            self.train_config.result_path if 'result_path' in self.train_config else None
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
            
            df = train_and_save_node2vec_model(
                train_dataset,
                noised,
                self.model_config,
                self.save_dir,
                self.test_config.embedding_path if 'embedding_path' in self.test_config else None,
                self.test_config.result_path if 'result_path' in self.test_config else None,
                save_paths={
                    'results': 'noise_results',
                    'embeddings': 'noise_embeddings'
                }
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
            train_dataset2 = eval(self.dataset_config.loader_name)(self.dataset_config, self.test_config.test2)
            
            df1 = train_and_save_node2vec_model(
                train_dataset1,
                train_dataset1.corr().fillna(1),
                self.model_config,
                self.save_dir,
                self.test_config.test1.embedding_path if 'embedding_path' in self.test_config.test1 else None.
                self.test_config.test1.result_path if 'result_path' in self.test_config.test1 else None,
                save_paths={
                    'results': 'results1',
                    'embeddings': 'embeddings1'
                }
            )
            
            df2 = train_and_save_node2vec_model(
                train_dataset2,
                train_dataset2.corr().fillna(1),
                self.model_config,
                self.save_dir,
                self.test_config.test2.embedding_path if 'embedding_path' in self.test_config.test2 else None.
                self.test_config.test2.result_path if 'result_path' in self.test_config.test2 else None,
                save_paths={
                    'results': 'results2',
                    'embeddings': 'embeddings2'
                }
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
