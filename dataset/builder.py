'''
Author: Thyssen Wen
Date: 2022-04-27 15:27:42
LastEditors: Thyssen Wen
LastEditTime: 2022-04-27 15:39:53
Description: dataset builder
FilePath: /ETESVS/dataset/builder.py
'''
from utils.build import Registry
from utils.build import build

DATASET = Registry('dataset')
PIPLINE = Registry('pipline')

def build_dataset(cfg):
    """Build dataset."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in DATASET:
        return build(cfg, DATASET)
    raise ValueError(f'{obj_type} is not registered in '
                     'DATASET')

def build_pipline(cfg):
    """Build dataset."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in PIPLINE:
        return build(cfg, PIPLINE)
    raise ValueError(f'{obj_type} is not registered in '
                     'PIPLINE')