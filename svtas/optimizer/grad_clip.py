'''
Author       : Thyssen Wen
Date         : 2022-11-21 10:53:11
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 10:55:32
Description  : file content
FilePath     : /SVTAS/svtas/optimizer/grad_clip.py
'''
from .builder import OPTIMIZER
import torch

@OPTIMIZER.register()
class GradClip(object):
    def __init__(self,
                 max_norm=40,
                 norm_type=2,
                 **kwargs) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, param):
        torch.nn.utils.clip_grad_norm_(parameters=param, max_norm=self.max_norm, norm_type=self.norm_type)