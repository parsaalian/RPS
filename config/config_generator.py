import yaml

base_dict = {
    "test": {
        "start_date": 201,
        "end_date": 290,
        "test_method": "future_performance"
    },
    "train": {
        "start_date": 0,
        "end_date": 200,
    },
    "runner": "RandomRunner",
    "exp_dir": "exp/Random",
    "model": {
        "name": "Random",
        "weight_method": "HRP_weight",
        "port_mean": 10,
        "port_std": 2,
        "count": 10,
    },
    "dataset": {
        "data_path": "data/indextrack/indextrack3.txt",
        "name": "indextrack3",
        "loader_name": "indextrack_loader"
    }
}

list_weight_methods = ["HRP_weight", 'MVO_weight', 'uniform_weight_returns']
list_MVO_model_configs = ['volatility', 'sharpe', 'risk', 'return']


def create_RPS_configs():
    for weight_method in list_weight_methods:
        base_dict['model']['weight_method'] = weight_method

        if weight_method == 'MVO_weight':
            for model_config in list_MVO_model_configs:
                base_dict['model']['model_config'] = model_config

                file_name = "Random_indextrack3_{weight_method}_{model_config}.yaml".format(
                    weight_method=weight_method, model_config=model_config)

                with open('./random/' + file_name, 'w') as outfile:
                    yaml.dump(base_dict, outfile, default_flow_style=False)

        else:
            file_name = "Random_indextrack3_{weight_method}.yaml".format(
                weight_method=weight_method)
            with open('./random/'+file_name, 'w') as outfile:
                yaml.dump(base_dict, outfile, default_flow_style=False)

            print(file_name)


create_RPS_configs()