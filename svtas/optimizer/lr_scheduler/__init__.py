'''
Author       : Thyssen Wen
Date         : 2022-10-31 14:44:09
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:45:18
Description  : file content
FilePath     : /SVTAS/svtas/optimizer/lr_scheduler/__init__.py
'''
from .multistep_lr import MultiStepLR
from .cosine_annealing_lr import CosineAnnealingLR

__all__ = [
    'MultiStepLR', 'CosineAnnealingLR'
]