'''
Author       : Thyssen Wen
Date         : 2023-10-23 20:01:04
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-23 20:08:19
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/tvm_model.py
'''
import numpy as np
from typing import Any, Dict, Sequence, List
from .base import BaseModel
from svtas.utils import AbstractBuildFactory, is_tvm_available

if is_tvm_available():
    from tvm.driver import tvmc
    from tvm import relay

@AbstractBuildFactory.register('model')
class TVMModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()