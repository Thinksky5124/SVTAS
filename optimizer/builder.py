'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:12:45
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 15:45:31
Description  : optimizer and learning rate scheduler builder
FilePath     : /ETESVS/optimizer/builder.py
'''
from utils.build import Registry
from utils.build import build

OPTIMIZER = Registry('optimizer')
LRSCHEDULER = Registry('lr_scheduler')

def build_optimizer(cfg):
    """Build optimizer."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in OPTIMIZER:
        return build(cfg, OPTIMIZER)
    raise ValueError(f'{obj_type} is not registered in '
                     'OPTIMIZER')

def build_lr_scheduler(cfg):
    """Build lr_scheduler."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in LRSCHEDULER:
        return build(cfg, LRSCHEDULER)
    raise ValueError(f'{obj_type} is not registered in '
                     'lr_scheduler')