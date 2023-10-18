'''
Author       : Thyssen Wen
Date         : 2023-10-18 15:10:23
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:23:45
Description  : file content
FilePath     : /SVTAS/svtas/profiling/base_profiler.py
'''
import abc
from typing import Dict
from svtas.utils.logger import BaseLogger

class BaseProfiler(metaclass=abc.ABCMeta):
    def __init__(self,
                 profile_step: int = 1) -> None:
        self.profile_step = profile_step
    
    @abc.abstractmethod
    def init_profiler(self, model, *args, **kwargs):
        pass

    @abc.abstractmethod
    def shutdown_profiler(self):
        pass

    @abc.abstractmethod
    def start_profile(self) -> None:
        pass
    
    @abc.abstractmethod
    def end_profile(self) -> None:
        pass
    
    @abc.abstractmethod
    def print_model_profile(self) -> None:
        pass