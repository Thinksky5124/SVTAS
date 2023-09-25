'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:37:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-22 16:38:05
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/base_iter_method.py
'''
import abc
from typing import Any

class BaseIterMethod(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("You must implement __call__ function!")