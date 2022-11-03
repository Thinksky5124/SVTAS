'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:10:49
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 16:48:06
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
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineLR

__all__ = [
    'MultiStepLR',
    'SGDOptimizer', 'TSMSGDOptimizer',
    'AdamOptimizer', 'TSMAdamOptimizer',
    'AdanOptimizer', 'AdamWOptimizer',
    'CosineAnnealingLR', 'WarmupMultiStepLR',
    'WarmupCosineLR'
]