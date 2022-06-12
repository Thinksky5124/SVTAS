'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:11:07
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-11 11:13:51
Description  : Recognition Framewwork
FilePath     : /ETESVS/model/architectures/recognition/__init__.py
'''
from .recognition2d import Recognition2D
from .recognition3d import Recognition3D

__all__ = [
    'Recognition2D', 'Recognition3D',
]