'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:10:35
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 19:14:21
Description  : AdamW optimizer
FilePath     : /SVTAS/svtas/optimizer/optim/adamw_optimizer.py
'''
from svtas.utils import AbstractBuildFactory
import torch
from .base_optim import TorchOptimizer

@AbstractBuildFactory.register('optimizer')
class AdamWOptimizer(TorchOptimizer, torch.optim.AdamW):
    def __init__(self,
                 model,
                 learning_rate=0.001,
                 betas=(0.9, 0.999),
                 weight_decay=0.01,
                 amsgrad=False,
                 maximize=False,
                 foreach=None,
                 capturable=False,
                 finetuning_scale_factor=0.1,
                 no_decay_key = [],
                 finetuning_key = [],
                 freeze_key = [],
                 **kwargs) -> None:
        params = self.get_optim_policies(model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay)
        super().__init__(params=params, lr=learning_rate, betas=betas,
                         weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, foreach=foreach, capturable=capturable)