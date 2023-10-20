'''
Author       : Thyssen Wen
Date         : 2023-10-19 16:43:49
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 18:50:16
Description  : file content
FilePath     : /SVTAS/svtas/inference/convertor/onnx_convertor.py
'''
import os
import torch
from typing import Any, Dict
from .base_convertor import BaseModelConvertor
from svtas.utils import AbstractBuildFactory, mkdir
from svtas.utils.logger import get_root_logger_instance

@AbstractBuildFactory.register('model_convertor')
class ONNXModelConvertor(BaseModelConvertor):
    def __init__(self,
                 input_names=['input_data'],
                 output_names=['output'],
                 dynamic_axes=None,
                 export_path: str = None) -> None:
        super().__init__(export_path)
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        try:
            import onnx
        except:
            raise ImportError("You must install onnx module to use ONNXModelConvertor!")
    
    def init_convertor(self):
        return super().init_convertor()
    
    def shutdown(self):
        return super().shutdown()

    def export(self, model: Any, data: Dict[str, Any], file_name: str, export_path: str = None):
        logger = get_root_logger_instance()
        logger.info("Start exporting ONNX model!")

        if export_path is None:
            export_path = os.path.join(self.export_path, "onnx")
        mkdir(export_path)
        export_full_path = os.path.join(export_path, file_name)

        input_data = dict(input_data = data)
        torch.onnx.export(model,
                          input_data,
                          export_full_path,
                          input_names=self.input_names,
                          output_names=self.output_names,
                          dynamic_axes=self.dynamic_axes)
        logger.info("Finish exporting ONNX model to " + export_path + " !")