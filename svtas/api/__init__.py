'''
Author       : Thyssen Wen
Date         : 2023-09-14 19:46:29
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 15:51:34
Description  : file content
FilePath     : /SVTAS/svtas/tasks/__init__.py
'''
from .infer import infer
from .export import export
from .extract import extract
from .profile import profile
from .test import test
from .train import train
from .visualize import visualize
from .visualize_loss import visulize_loss

__all__ = [
    "export", "extract", "profile", "test", "train", "visualize", "visulize_loss",
    "infer"
]