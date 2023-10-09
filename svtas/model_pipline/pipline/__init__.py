'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:13:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 09:27:29
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/__init__.py
'''
from .base_pipline import BaseModelPipline, FakeModelPipline
from .torch_model_pipline import TorchModelPipline
from .deepspeed_model_pipline import DeepspeedModelPipline
from .torch_model_ddp_pipline import TorchDistributedDataParallelModelPipline
from .torch_cam_model_pipline import TorchCAMModelPipline
__all__ = [
    'BaseModelPipline', 'TorchModelPipline', 'DeepspeedModelPipline',
    'TorchDistributedDataParallelModelPipline', 'FakeModelPipline',
    'TorchCAMModelPipline'
]