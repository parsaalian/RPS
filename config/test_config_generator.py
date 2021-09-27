import yaml
import os
from easydict import EasyDict as edict

def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj

def create_test_config(base_dir, save_dir, result_name, test_sets):
    for folder in os.listdir(base_dir):
        config_path = os.path.join(base_dir, folder, 'config.yaml')
        print(config_path)
        config = edict(yaml.load(open(config_path, 'r')))
        tests = test_sets['indextrack'] if 'indextrack' in folder else test_sets['sp500']
                
        for test in tests:
            '''config.test = {
                **test,
                "train_results": os.path.join(base_dir.replace('../', ''), folder, result_name)
            }'''
            config.test = test
            # print(config.model, '\n')
            yaml.dump(
                edict2dict(config),
                open(os.path.join(
                    save_dir, '_'.join(folder.split('_')[:-1]) + \
                        (('_' + config.model.model_config) if 'model_config' in config.model else '') + \
                    '_' + config.test.test_method + '_test.yaml'
                ), 'w+'),
                default_flow_style=False
            )

sp500_test_sets = [
    {
        "test_method": "future_performance",
        "start_date": '2019-08-02',
        "end_date": '2019-09-01'
    },
    {
        "test_method": "noise_stability",
        "sort_column": "sharpe",
        "noise_sigma": 0.01,
        "start_date": '2019-08-02',
        "end_date": '2019-09-01'
    },
    {
        "test_method": "time_stability",
        "sort_column": "sharpe",
        "test1": {
            "start_date": '2019-04-01',
            "end_date": '2019-04-08'
        },
        "test2": {
            "start_date": '2019-04-09',
            "end_date": '2019-08-16'
        }
    }
]

indextrack_test_sets = [
    {
        "test_method": "future_performance",
        "start_date": 201,
        "end_date": 290
    },
    {
        "test_method": "noise_stability",
        "sort_column": "sharpe",
        "noise_sigma": 0.01,
        "start_date": 201,
        "end_date": 290
    },
    {
        "test_method": "time_stability",
        "sort_column": "sharpe",
        "test1": {
            "start_date": 0,
            "end_date": 20
        },
        "test2": {
            "start_date": 20,
            "end_date": 40
        }
    }
]

create_test_config(
    '../exp/Mantegna/train',
    './mantegna/test',
    'mantegna_results.csv',
    {
        'sp500': sp500_test_sets[1:],
        'indextrack': indextrack_test_sets[1:]
    }
)