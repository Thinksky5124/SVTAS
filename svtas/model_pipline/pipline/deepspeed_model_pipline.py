'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:27:09
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-21 19:27:16
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/deepspeed_model_pipline.py
'''
from .torch_model_pipline import TorchModelPipline
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model_pipline')
class DeepspeedModelPipline(TorchModelPipline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)