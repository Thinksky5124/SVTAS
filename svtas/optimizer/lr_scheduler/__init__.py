'''
Author       : Thyssen Wen
Date         : 2022-10-31 14:44:09
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 21:01:39
Description  : file content
FilePath     : /SVTAS/svtas/optimizer/lr_scheduler/__init__.py
'''
from .base_lr_scheduler import BaseLRScheduler, TorchLRScheduler
from .multistep_lr import MultiStepLR
from .cosine_annealing_lr import CosineAnnealingLR
from .cosin_warmup_lr import WarmupCosineLR
from .multistep_warmup_lr import WarmupMultiStepLR
from .cosine_annealing_warmup_restart_lr import CosineAnnealingWarmupRestarts

__all__ = [
    'MultiStepLR', 'CosineAnnealingLR', 'WarmupCosineLR', 'WarmupMultiStepLR',
    'CosineAnnealingWarmupRestarts', 'BaseLRScheduler', 'TorchLRScheduler'
]