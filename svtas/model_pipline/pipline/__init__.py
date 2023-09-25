'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:13:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-22 13:54:24
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/__init__.py
'''
from .base_pipline import BaseModelPipline
from .torch_model_pipline import TorchModelPipline
from .deepspeed_model_pipline import DeepspeedModelPipline

__all__ = [
    'BaseModelPipline', 'TorchModelPipline', 'DeepspeedModelPipline'
]