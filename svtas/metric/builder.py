'''
Author: Thyssen Wen
Date: 2022-04-27 15:27:42
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 15:09:37
Description: metric builder
FilePath     : /ETESVS/metric/builder.py
'''
from utils.build import Registry
from utils.build import build

METRIC = Registry('metric')

def build_metric(cfg):
    """Build metric."""
    args = cfg.copy()
    obj_type = args.get('name')
    if obj_type in METRIC:
        return build(cfg, METRIC)
    raise ValueError(f'{obj_type} is not registered in '
                     'METRIC')