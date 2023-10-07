'''
Author       : Thyssen Wen
Date         : 2023-10-05 19:15:04
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-06 16:37:45
Description  : file content
FilePath     : /SVTAS/svtas/optimizer/lr_scheduler/base_lr_scheduler.py
'''
import abc
from torch.optim.lr_scheduler import _LRScheduler

class BaseLRScheduler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_lr(self):
        pass
    
    @abc.abstractmethod
    def step(self):
        pass

class TorchLRScheduler(_LRScheduler, BaseLRScheduler):
    pass