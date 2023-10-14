'''
Author       : Thyssen Wen
Date         : 2023-10-12 15:34:26
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-12 15:34:34
Description  : file content
FilePath     : /SVTAS/svtas/model/unet/base_unet.py
'''
import torch
import torch.nn as nn
from .condition_unet import ConditionUnet

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class ConditionUnet1D(ConditionUnet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)