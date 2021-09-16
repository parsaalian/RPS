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
    config.exp_name = '_'.join([
        config.model.name, config.dataset.name,
        time.strftime('%Y-%b-%d-%H-%M-%S')
    ])

    if exp_dir is not None:
        config.exp_dir = exp_dir
    
    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
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