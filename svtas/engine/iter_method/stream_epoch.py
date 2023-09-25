'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:41:13
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 10:44:02
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/stream_epoch.py
'''
from typing import Any
from .base_iter_method import BaseIterMethod
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('iter_method')
class StreamWithValidationEpochMethod(BaseIterMethod):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)