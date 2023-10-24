'''
Author       : Thyssen Wen
Date         : 2023-10-23 20:09:31
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-23 21:02:15
Description  : file content
FilePath     : /SVTAS/svtas/inference/convertor/tvm_convertor.py
'''
import os
import torch
from typing import Any, Dict, List
from .onnx_convertor import ONNXModelConvertor
from svtas.utils import AbstractBuildFactory, mkdir, is_tvm_available
from svtas.utils.logger import get_root_logger_instance

# if is_tvm_available():
from tvm.driver import tvmc
from tvm import relay

@AbstractBuildFactory.register('model_convertor')
class TVMModelConvertor(ONNXModelConvertor):
    def __init__(self,
                 target: str='llvm',
                 input_names: List[str]=['input_data'],
                 output_names: List[str]=['output'],
                 shape_optim_list: Dict[str, List[List[int]]] = None,
                 dynamic_axes: List[int]=None,
                 export_path: str = None,
                 opset_version: int = None) -> None:
        super().__init__(input_names, output_names, dynamic_axes,
                         export_path, opset_version)
        self.target = target
        self.shape_optim_list = shape_optim_list

    def export(self, model: Any, data: Dict[str, torch.Tensor], file_name: str, export_path: str = None):
        logger = get_root_logger_instance()
        logger.info("Start exporting TensorRT model!\n Step 1: PyTorch Model convert to ONNX Format!")
        # step 1.torch -> onnx
        super().export(model, data, "temp", export_path)
        # step 2. onnx -> tvm
        if export_path is None:
            export_onnx_path = os.path.join(self.export_path, "onnx")
            export_tvm_path = os.path.join(self.export_path, "tvm")
        mkdir(export_tvm_path)
        onnx_file_path = os.path.join(export_onnx_path, "temp.onnx")
        tvm_file_path = os.path.join(export_tvm_path, file_name + '.so')

        logger.info("Start export ONNX file to TVM model library...")
        # load onnx file
        model = tvmc.load(onnx_file_path)
        # compile onnx model to tvm realy model
        package = tvmc.compile(model, target=self.target, package_path=tvm_file_path)
        logger.info("Finish exporting TensorRT model to " + tvm_file_path + " !")