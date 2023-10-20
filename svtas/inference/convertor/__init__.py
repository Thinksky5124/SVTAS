'''
Author       : Thyssen Wen
Date         : 2023-10-19 16:39:42
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 16:45:18
Description  : file content
FilePath     : /SVTAS/svtas/inference/convertor/__init__.py
'''
from .base_convertor import BaseModelConvertor
from .onnx_convertor import ONNXModelConvertor

__all__ = [
    'BaseModelConvertor', 'ONNXModelConvertor'
]