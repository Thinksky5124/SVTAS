'''
Author       : Thyssen Wen
Date         : 2022-11-21 19:46:24
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 20:11:35
Description  : file content
FilePath     : /SVTAS/svtas/model/backbones/utils/efficientnet/__init__.py
'''
from .functions import make_divisible
from .layers import SqueezeExcitation, InvertedResidual, EdgeResidual
__all__ = [
    "make_divisible", "SqueezeExcitation", "InvertedResidual",
    "EdgeResidual"
]