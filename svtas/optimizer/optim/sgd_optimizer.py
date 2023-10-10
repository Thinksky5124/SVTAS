'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:11:25
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-27 21:07:01
Description  : SGD optimizer
FilePath     : /SVTAS/svtas/optimizer/optim/sgd_optimizer.py
'''
from svtas.utils import AbstractBuildFactory
import torch
from .base_optim import TorchOptimizer

@AbstractBuildFactory.register('optimizer')
class SGDOptimizer(TorchOptimizer, torch.optim.SGD):
    def __init__(self,
                 model,
                 learning_rate=0.01,
                 momentum=0.9,
                 weight_decay=1e-4,
                 finetuning_scale_factor=0.1,
                 no_decay_key = [],
                 finetuning_key = [],
                 freeze_key = [],
                 **kwargs) -> None:
        params = self.get_optim_policies(model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay)
        super().__init__(params=params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)