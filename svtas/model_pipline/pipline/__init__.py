'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:13:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 16:58:48
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/__init__.py
'''
from .base_pipline import BaseModelPipline
from .torch import *
from .onnx_runtime import *
from .openvino import *
from .tensorrt import *