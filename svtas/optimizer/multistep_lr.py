'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:27:58
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 15:38:38
Description  : multi-step learning rate schedular
FilePath     : /ETESVS/optimizer/multistep_lr.py
'''
from .builder import LRSCHEDULER
import torch

@LRSCHEDULER.register()
class MultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self,
                 optimizer,
                 step_size=[10, 30],
                 gamma=0.1) -> None:
        super().__init__(optimizer=optimizer, milestones=step_size, gamma=gamma)