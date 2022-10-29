'''
Author       : Thyssen Wen
Date         : 2022-10-28 19:50:48
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 19:57:59
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/segmentation/video/__init__.py
'''
from .segmentation2d import Segmentation2D
from .segmentation3d import Segmentation3D

__all__ = [
    "Segmentation2D", "Segmentation3D"
]