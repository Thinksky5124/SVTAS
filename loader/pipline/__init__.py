'''
Author       : Thyssen Wen
Date         : 2022-05-18 14:55:31
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 19:55:40
Description  : Pipline function
FilePath     : /ETESVS/loader/pipline/__init__.py
'''
from .collect_fn import BatchCompose, StreamBatchCompose
from .base_pipline import BasePipline

__all__ = [
    'BatchCompose', 'BasePipline', 'StreamBatchCompose'
]