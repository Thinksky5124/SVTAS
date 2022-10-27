'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:11:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 16:08:05
Description  : SGD optimizer
FilePath     : /ETESVS/optimizer/sgd_optimizer.py
'''
from .builder import OPTIMIZER
import torch

@OPTIMIZER.register()
class SGDOptimizer(torch.optim.SGD):
    def __init__(self,
                 model,
                 learning_rate=0.01,
                 momentum=0.9,
                 weight_decay=1e-4) -> None:
        super().__init__(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)