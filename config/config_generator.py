import yaml

base_dict = {
    "test": {},
    "train": {
        "start_date": 0,
        "end_date": 200,
    },
    "runner": "MantegnaRunner",
    "exp_dir": "exp/Mantegna",
    "model": {
        "name": "Mantegna",
        "weight_method": "HRP_weight",
    },
    "dataset": {
        "data_path": "data/indextrack/indextrack5.txt",
        "name": "indextrack5",
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

                file_name = "Mantegna_indextrack5_{weight_method}_{model_config}.yaml".format(
                    weight_method=weight_method, model_config=model_config)

                with open('./mantegna/' + file_name, 'w') as outfile:
                    yaml.dump(base_dict, outfile, default_flow_style=False)

        else:
            file_name = "Mantegna_indextrack5_{weight_method}.yaml".format(
                weight_method=weight_method)
            with open('./mantegna/'+file_name, 'w') as outfile:
                yaml.dump(base_dict, outfile, default_flow_style=False)

            print(file_name)


create_RPS_configs()