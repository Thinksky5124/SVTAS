'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:16:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 15:18:52
Description  : Metric Base class
FilePath     : /ETESVS/metric/base_metric.py
'''
import abc
from .builder import METRIC

@METRIC.register()
class BaseMetric(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, outputs):
        """update metrics during each iter
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        raise NotImplementedError