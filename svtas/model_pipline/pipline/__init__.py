'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:13:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-29 11:28:41
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/__init__.py
'''
from .torch import *
from .base_pipline import BaseModelPipline
from .onnxruntime_model_pipline import ONNXRuntimeModelPipline
from .trt_model_pipline import TensorRTModelPipline