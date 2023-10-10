'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:16:35
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 11:18:46
Description  : Metric Base class
FilePath     : /SVTAS/svtas/metric/base_metric.py
'''
import abc

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