'''
Author       : Thyssen Wen
Date         : 2023-09-21 20:32:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-21 20:35:24
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/base.py
'''
import abc
from typing import Any

class BaseWapper(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("You must implement __call__ function!")