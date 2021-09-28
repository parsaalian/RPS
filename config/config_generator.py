import yaml

base_dict = {
    "test": {},
    "train": {
        "start_date": 0,
        "end_date": 200,
    },
    "runner": "SplexRunner",
    "exp_dir": "exp/Splex",
    "model": {
        "name": "Splex",
        "weight_method": "HRP_weight",
        "weight_limit": 0.2,
        "s": 2,
    },
    "dataset": {
        "data_path": "data/SP500/SP_20180402_20200401.csv",
        "name": "sp500",
        "loader_name": "SP500_loader"
    }
}

list_weight_methods = ["HRP_weight", "CLA_weight", 'MVO_weight', 'uniform_weight_returns']
list_MVO_model_configs = ['volatility', 'sharpe', 'risk', 'return']
list_CLA_model_configs = ['volatility', 'sharpe']


def create_RPS_configs():
    for weight_method in list_weight_methods:
        base_dict['model']['weight_method'] = weight_method

        if weight_method == 'MVO_weight':
            for model_config in list_MVO_model_configs:
                base_dict['model']['model_config'] = model_config

                file_name = "Splex_sp500_{weight_method}_{model_config}.yaml".format(
                    weight_method=weight_method, model_config=model_config)

                with open('./splex/' + file_name, 'w') as outfile:
                    yaml.dump(base_dict, outfile, default_flow_style=False)
        
        elif weight_method == 'CLA_weight':
            for model_config in list_CLA_model_configs:
                base_dict['model']['model_config'] = model_config

                file_name = "Splex_sp500_{weight_method}_{model_config}.yaml".format(
                    weight_method=weight_method, model_config=model_config)

                with open('./splex/' + file_name, 'w') as outfile:
                    yaml.dump(base_dict, outfile, default_flow_style=False)

        else:
            file_name = "Splex_sp500_{weight_method}.yaml".format(
                weight_method=weight_method)
            with open('./splex/'+file_name, 'w') as outfile:
                yaml.dump(base_dict, outfile, default_flow_style=False)


create_RPS_configs()