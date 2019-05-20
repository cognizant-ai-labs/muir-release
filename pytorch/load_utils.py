#
# Utils for loading.
#

import yaml
from datasets.load_dataset import load_dataset
from models.create_net import create_net

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config

def load_tasks(tasks_config):
    return [load_task(task_config) for task_config in task_configs]

def load_task(task_config):
    dataset = load_dataset(tasks_config['dataset'])
    model = create_model(tasks_config['model'])
    return (dataset, model)

