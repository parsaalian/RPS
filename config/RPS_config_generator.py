import yaml

base_dict = {
    "test": None,
    "train": {
        "start_date": "2019-04-01",
        "end_date": "2019-08-01",
        "embedding_path": "exp/RPS/embeddings.npy"
    },
    "runner": "RPSRunner",
    "exp_dir": "exp/RPS",
    "model": {
        "walk_length": 2,
        "num_walks": 50,
        "name": "RPS",
        "n_clusters": 40,
        "clustering_method": "FCM_clustering",
        "workers": 7,
        "distance_transformer": "coth_transformer",
        "window": 10,
        "cluster_method": "fuzzy_c_means",
        "weight_method": "HRP_weight",
        "model_config": None,
        "max_memberships": 2,
        "batch_words": 4,
        "min_count": 1,
        "dimensions": 64,
        "sort_measure": "sharpe"
    },
    "dataset": {
        "data_path": "data/SP500/SP_20180402_20200401.csv",
        "name": "sp500",
        "loader_name": "SP500_loader"
    }
}

list_clustering_methods = ["FCM_clustering", "KMEANS_clustering"]
list_n_clusters = [40, 50, 60]
# list_weight_methods = ["HRP_weight", 'MVO_weight', 'uniform_weight_returns']
list_weight_methods = ["CLA_weights"]
list_MVO_model_configs = ['volatility', 'sharpe', 'risk', 'return']


def create_RPS_configs():
    for n_cluster in list_n_clusters:
        for clustering_method in list_clustering_methods:
            for weight_method in list_weight_methods:
                base_dict['model']['n_clusters'] = n_cluster
                base_dict['model']['clustering_method'] = clustering_method
                base_dict['model']['weight_method'] = weight_method

                if weight_method == 'MVO_weight':
                    for model_config in list_MVO_model_configs:
                        base_dict['model']['model_config'] = model_config

                        file_name = "RPS_{clustering_method}_{n_cluster}_{weight_method}_{model_config}.yaml".format(
                            clustering_method=clustering_method, n_cluster=n_cluster,
                            weight_method=weight_method, model_config=model_config)

                        with open('./rps/sp500/' + file_name, 'w') as outfile:
                            yaml.dump(base_dict, outfile, default_flow_style=False)

                else:
                    file_name = "RPS_{clustering_method}_{n_cluster}_{weight_method}.yaml".format(
                        clustering_method=clustering_method, n_cluster=n_cluster, weight_method=weight_method)
                    with open('./rps/sp500/'+file_name, 'w') as outfile:
                        yaml.dump(base_dict, outfile, default_flow_style=False)

                    print(file_name)


create_RPS_configs()