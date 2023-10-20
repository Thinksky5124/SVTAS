'''
Author       : Thyssen Wen
Date         : 2023-10-19 18:56:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 16:58:01
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch/__init__.py
'''
from .base import BaseTorchModelPipline, FakeTorchModelPipline
from .torch_model_pipline import TorchModelPipline
from .deepspeed_model_pipline import DeepspeedModelPipline
from .torch_model_ddp_pipline import TorchDistributedDataParallelModelPipline
from .torch_cam_model_pipline import TorchCAMModelPipline
__all__ = [
    'TorchModelPipline', 'DeepspeedModelPipline',
    'TorchDistributedDataParallelModelPipline',
    'TorchCAMModelPipline', 'BaseTorchModelPipline',
    'FakeTorchModelPipline'
]