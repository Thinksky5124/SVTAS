'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:40:18
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 10:43:17
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/epoch.py
'''
from typing import Any
from .base_iter_method import BaseIterMethod
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('iter_method')
class EpochWithValidationMethod(BaseIterMethod):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)