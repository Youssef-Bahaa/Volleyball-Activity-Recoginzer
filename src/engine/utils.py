import torch.optim as optim
import torch.optim.lr_scheduler
import yaml
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_optimizer(cfg: dict, model):
    lr = cfg['training']['learning_rate']
    weight_decay = cfg['training'].get('weight_decay', 0.0)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



def build_scheduler(cfg: dict, optimizer):
    min_lr = float(cfg['training']['min_lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=cfg['training']['scheduler_mode'], factor=cfg['training']['scheduler_factor'],
        patience=cfg['training']['scheduler_patience'], min_lr=min_lr
    )
    return scheduler






