import torch


def load_fairseq_model(pkg):
    """
    pkg: dict_keys(['args', 'model', 'optimizer_history', 'extra_state', 'last_optimizer_state'])
    
    """
