'''
Author       : Thyssen Wen
Date         : 2023-10-19 18:58:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 21:53:24
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/onnx_runtime/onnxruntime_model_pipline.py
'''
import torch
import numpy as np
from typing import Any, Dict
from ..base_pipline import BaseModelPipline, BaseInferModelPipline
from svtas.utils import AbstractBuildFactory, get_logger
from svtas.utils import is_onnx_available
from ...wrapper import ONNXRuntimeModel

if is_onnx_available():
    import onnx
    import onnxruntime as ort

@AbstractBuildFactory.register('model_pipline')
class ONNXRuntimeModelPipline(BaseInferModelPipline):
    def __init__(self,
                model: str | ONNXRuntimeModel,
                post_processing: Dict,
                device = None) -> None:
        super().__init__(model, post_processing, device)
        if isinstance(model, ONNXRuntimeModel):
            self.model = model
    
    def load_from_ckpt_file(self, ckpt_path: str = None):
        ckpt_path = self.load_from_ckpt_file_ckeck(ckpt_path)
        model = onnx.load(ckpt_path)
        onnx.checker.check_model(model)
        self.model = ort.InferenceSession(ckpt_path)
