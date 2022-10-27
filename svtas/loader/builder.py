'''
Author: Thyssen Wen
Date: 2022-04-27 15:27:42
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 15:25:12
Description: dataset builder
FilePath     : /ETESVS/loader/builder.py
'''
from utils.build import Registry
from utils.build import build

DATASET = Registry('dataset')
PIPLINE = Registry('pipline')
DECODE = Registry('decode')
SAMPLER = Registry('sampler')
TRANSFORM = Registry('transform')

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

def build_decode(cfg):
    """Build decode."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in DECODE:
        return build(cfg, DECODE)
    raise ValueError(f'{obj_type} is not registered in '
                     'DECODE')

def build_sampler(cfg):
    """Build sampler."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in SAMPLER:
        return build(cfg, SAMPLER)
    raise ValueError(f'{obj_type} is not registered in '
                     'SAMPLER')

def build_transform(cfg):
    """Build transform."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in TRANSFORM:
        return build(cfg, TRANSFORM)
    raise ValueError(f'{obj_type} is not registered in '
                     'TRANSFORM')