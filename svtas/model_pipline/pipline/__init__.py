'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:13:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 16:43:48
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/__init__.py
'''
from .base_pipline import BaseModelPipline
from .torch import *
from .onnxruntime_model_pipline import ONNXRuntimeModelPipline
from .trt_model_pipline import TensorRTModelPipline