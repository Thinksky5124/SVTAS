'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:13:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-06 15:18:21
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/__init__.py
'''
from .base_pipline import BaseModelPipline, FakeModelPipline
from .torch_model_pipline import TorchModelPipline
from .deepspeed_model_pipline import DeepspeedModelPipline
from .torch_model_ddp_pipline import TorchDDPModelPipline, TorchFSDPModelPipline

__all__ = [
    'BaseModelPipline', 'TorchModelPipline', 'DeepspeedModelPipline',
    'TorchDDPModelPipline', 'TorchFSDPModelPipline', 'FakeModelPipline'
]