'''
Author       : Thyssen Wen
Date         : 2022-05-06 16:05:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 16:06:07
Description  : adam optimizer
FilePath     : /ETESVS/optimizer/adam_optimizer.py
'''
from .builder import OPTIMIZER
import torch

@OPTIMIZER.register()
class AdamOptimizer(torch.optim.Adam):
    def __init__(self,
                 model,
                 learning_rate=0.01,
                 betas=(0.9, 0.999),
                 weight_decay=1e-4) -> None:
        super().__init__(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=betas, weight_decay=weight_decay)