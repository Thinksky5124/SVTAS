'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:30
LastEditors: Thyssen Wen
LastEditTime: 2022-04-29 11:03:22
Description: file content
FilePath: /ETESVS/model/architectures/__init__.py
'''
from .etesvs import ETESVS
from .feature_segmentation import FeatureSegmentation
from .recognition2d import Recognition2D

__all__ = [
    'ETESVS', 'FeatureSegmentation', 'Recognition2D'
]