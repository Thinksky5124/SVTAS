'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:10:49
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 19:18:26
Description  : optimizer module
FilePath     : /SVTAS/svtas/optimizer/__init__.py
'''
from .lr_scheduler.multistep_lr import MultiStepLR
from .optim.sgd_optimizer import SGDOptimizer
from .optim.adam_optimizer import AdamOptimizer
from .optim.tsm_sgd_optimizer import TSMSGDOptimizer
from .optim.tsm_adam_optimizer import TSMAdamOptimizer
from .optim.adan_optimizer import AdanOptimizer
from .optim.adamw_optimizer import AdamWOptimizer
from .lr_scheduler.cosine_annealing_lr import CosineAnnealingLR
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineLR, CosineAnnealingWarmupRestarts
from .grad_clip import GradClip
from .lr_scheduler import BaseLRScheduler
from .optim import TorchOptimizer

__all__ = [
    'MultiStepLR',
    'SGDOptimizer', 'TSMSGDOptimizer',
    'AdamOptimizer', 'TSMAdamOptimizer',
    'AdanOptimizer', 'AdamWOptimizer',
    'CosineAnnealingLR', 'WarmupMultiStepLR',
    'WarmupCosineLR', 'CosineAnnealingWarmupRestarts',
    'GradClip', 'BaseLRScheduler', 'TorchOptimizer'
]