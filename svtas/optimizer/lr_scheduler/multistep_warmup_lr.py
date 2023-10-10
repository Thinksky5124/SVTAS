'''
Author       : Thyssen Wen
Date         : 2022-11-03 16:43:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 21:03:22
Description  : ref:https://github.com/amazon-science/long-short-term-transformer
FilePath     : /SVTAS/svtas/optimizer/lr_scheduler/multistep_warmup_lr.py
'''

from bisect import bisect_left

from svtas.utils import AbstractBuildFactory
from .base_lr_scheduler import TorchLRScheduler

def _get_warmup_factor_at_iter(warmup_method,
                               this_iter,
                               warmup_iters,
                               warmup_factor):
    if this_iter >= warmup_iters:
        return 1.0

    if warmup_method == 'constant':
        return warmup_factor
    elif warmup_method == 'linear':
        alpha = this_iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError('Unknown warmup method: {}'.format(warmup_method))

@AbstractBuildFactory.register('lr_scheduler')
class WarmupMultiStepLR(TorchLRScheduler):

    def __init__(self,
                 optimizer,
                 milestones=[],
                 gamma=0.1,
                 warmup_factor=0.3,
                 warmup_iters=500,
                 warmup_method='linear',
                 last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. Got {}'.format(milestones)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.last_epoch,
            self.warmup_iters,
            self.warmup_factor,
        )
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_left(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()

