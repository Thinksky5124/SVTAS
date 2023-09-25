'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:28:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-24 10:58:36
Description  : file content
FilePath     : /SVTAS/svtas/engine/base_engine.py
'''
import abc
from svtas.model_pipline import BaseModelPipline
from svtas.utils.logger import BaseLogger
from .iter_method import BaseIterMethod

class BaseEngine(metaclass=abc.ABCMeta):
    model_pipline: BaseModelPipline
    logger: BaseLogger
    iter_method: BaseIterMethod
    
    def __init__(self, *agrs, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def epoch_init(self):
        pass

    @abc.abstractmethod
    def epoch_end(self):
        pass

    @abc.abstractmethod
    def resume(self, resume_cfg_dict):
        pass
    
    @abc.abstractmethod
    def batch_end_step(self, sliding_num, vid_list, step):
        pass

    @abc.abstractmethod
    def _model_forward(self, data_dict):
        raise NotImplementedError("You must implement _model_forward function!")
    
    @abc.abstractmethod
    def run_one_clip(self, data_dict):
        pass
    
    @abc.abstractmethod
    def run_one_iter(self, data, r_tic=None, epoch=None):
        pass

    @abc.abstractmethod
    def run_one_batch(self, data, r_tic=None, epoch=None):
        pass
    
    @abc.abstractmethod
    def run():
        raise NotImplementedError("You must implement run function!")