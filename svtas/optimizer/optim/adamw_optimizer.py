'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:10:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 15:29:41
Description  : AdamW optimizer
FilePath     : /SVTAS/svtas/optimizer/optim/adamw_optimizer.py
'''
from ..builder import OPTIMIZER
import torch

@OPTIMIZER.register()
class AdamWOptimizer(torch.optim.AdamW):
    def __init__(self,
                 model,
                 learning_rate=0.001,
                 betas=(0.9, 0.999),
                 weight_decay=0.01,
                 amsgrad=False,
                 maximize=False,
                 foreach=None,
                 capturable=False,
                 **kwargs) -> None:
        super().__init__(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=betas,
                         weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, foreach=foreach, capturable=capturable)