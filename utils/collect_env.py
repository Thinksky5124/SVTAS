'''
Author       : Thyssen Wen
Date         : 2022-05-09 14:54:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-09 14:55:42
Description  : collect env info ref:https://github.com/open-mmlab/mmaction2/blob/master/mmaction/utils/collect_env.py
FilePath     : /ETESVS/utils/collect_env.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash


def collect_env():
    env_info = collect_basic_env()
    env_info['ETESVS'] = (get_git_hash(digits=7))
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')