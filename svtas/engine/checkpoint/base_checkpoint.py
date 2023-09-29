'''
Author       : Thyssen Wen
Date         : 2023-09-25 17:06:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 17:06:27
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/base_checkpoint.py
'''
import abc
from typing import Any

class BaseCheckpointor(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save(self, *args: Any, **kwds: Any) -> bool:
        raise NotImplementedError("You must implement __call__ function!")
    
    @abc.abstractmethod
    def load(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("You must implement __call__ function!")