'''
Author       : Thyssen Wen
Date         : 2023-09-21 20:32:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 19:27:34
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/base.py
'''
import abc
from typing import Any

class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def _clear_memory_buffer(self):
        pass
    
    @abc.abstractmethod
    def init_weights(self):
        pass
    
    @abc.abstractmethod
    def train(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement train function!")
    
    @abc.abstractmethod
    def infer(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement infer function!")
    
    @abc.abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        pass