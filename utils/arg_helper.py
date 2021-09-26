import os
import yaml
import time
import argparse
from easydict import EasyDict as edict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Running Experiments of RPS Model")
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default="config/rps.yaml",
        required=True,
        help="Path of config file"
    )
    parser.add_argument(
        '-l',
        '--log_level',
        type=str,
        default='INFO',
        help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL"
    )
    parser.add_argument('-t', '--test', help="Test model", action='store_true')
    args = parser.parse_args()
    return args


def get_config(config_file, exp_dir=None):
    """ Construct and snapshot hyper parameters """
    config = edict(yaml.load(open(config_file, 'r')))

    # create hyper parameters
    config.run_id = str(os.getpid())
    
    exp_name = list(filter(lambda x: x is not None, [
        config.model.name,
        config.dataset.name,
        config.model.clustering_method if 'clustering_method' in config.model else None,
        str(config.model.n_clusters) if 'n_clusters' in config.model else None,
        config.model.weight_method if 'weight_method' in config.model else None,
        config.model.optimize_method if 'optimize_method' in config.model else None,
        str(config.model.cardinality) if 'cardinality' in config.model else None,
        str(config.model.s) if 's' in config.model else None,
        time.strftime('%Y-%b-%d-%H-%M-%S'),
    ]))
    
    config.exp_name = '_'.join(exp_name)

    if exp_dir is not None:
        config.exp_dir = exp_dir
    
    if config.test:
        config.save_dir = os.path.join(config.exp_dir, 'test', config.exp_name, config.test.test_method)
    else:
        config.save_dir = os.path.join(config.exp_dir, 'train', config.exp_name)
    save_name = os.path.join(config.save_dir, 'config.yaml')

    mkdir(config.exp_dir)
    mkdir(config.save_dir)

    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

    return config


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)