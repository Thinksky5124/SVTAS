'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:30
LastEditors: Thyssen Wen
LastEditTime: 2022-04-27 19:59:12
Description: file content
FilePath: /ETESVS/model/architectures/__init__.py
'''
from .etesvs import ETESVS
from .feature_segmentation import FeatureSegmentation

__all__ = [
    'ETESVS', 'FeatureSegmentation'
]