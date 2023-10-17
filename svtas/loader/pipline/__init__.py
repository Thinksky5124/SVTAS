'''
Author       : Thyssen Wen
Date         : 2022-05-18 14:55:31
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 19:33:37
Description  : Pipline function
FilePath     : /SVTAS/svtas/loader/pipline/__init__.py
'''
from .collect_fn import BatchCompose
from .base_pipline import BaseDatasetPipline

__all__ = [
    'BatchCompose', 'BaseDatasetPipline'
]