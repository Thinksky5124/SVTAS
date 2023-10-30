'''
Author       : Thyssen Wen
Date         : 2023-10-30 15:22:22
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 15:25:36
Description  : file content
FilePath     : /SVTAS/svtas/serving/client/visualizer/__init__.py
'''
from .base import BaseClientViusalizer
from .opencv_visualizer import OpencvViusalizer

__all__ = [
    "BaseClientViusalizer", "OpencvViusalizer"
]