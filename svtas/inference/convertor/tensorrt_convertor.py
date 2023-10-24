'''
Author       : Thyssen Wen
Date         : 2023-10-21 20:23:49
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 15:12:58
Description  : file content
FilePath     : /SVTAS/svtas/inference/convertor/tensorrt_convertor.py
'''
import os
import torch
from typing import Any, Dict, List
from .onnx_convertor import ONNXModelConvertor
from svtas.utils import AbstractBuildFactory, mkdir, is_tensorrt_available
from svtas.utils.logger import get_root_logger_instance

if is_tensorrt_available():
    import tensorrt as trt
    from svtas.utils.logger import TensorRTLogger

@AbstractBuildFactory.register('model_convertor')
class TensorRTModelConvertor(ONNXModelConvertor):
    def __init__(self,
                 tensorrt_fp16_qat: bool = False,
                 tensorrt_int8_qat: bool = False,
                 input_names=['input_data'],
                 output_names=['output'],
                 shape_optim_list: Dict[str, List[List[int]]] = None,
                 dynamic_axes=None,
                 export_path: str = None,
                 opset_version: int = None,
                 int8_calibrator = None) -> None:
        super().__init__(input_names, output_names, dynamic_axes,
                         export_path, opset_version)
        self.tensorrt_fp16_qat = tensorrt_fp16_qat
        self.tensorrt_int8_qat = tensorrt_int8_qat
        self.int8_calibrator = int8_calibrator
        self.shape_optim_list = shape_optim_list

    def export(self, model: Any, data: Dict[str, torch.Tensor], file_name: str, export_path: str = None):
        logger = get_root_logger_instance()
        logger.info("Start exporting TensorRT model!\n Step 1: PyTorch Model convert to ONNX Format!")
        # step 1.torch -> onnx
        super().export(model, data, "temp", export_path)
        # step 2. onnx -> tensorrt
        if export_path is None:
            export_onnx_path = os.path.join(self.export_path, "onnx")
            export_trt_path = os.path.join(self.export_path, "trt")
        mkdir(export_trt_path)
        onnx_file_path = os.path.join(export_onnx_path, "temp.onnx")
        trt_file_path = os.path.join(export_trt_path, file_name + '.plan')

        logger.info("Start export ONNX file to TensorRT...")

        logger_rt = TensorRTLogger()
        builder = trt.Builder(logger_rt)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if self.tensorrt_fp16_qat:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.tensorrt_int8_qat:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = self.int8_calibrator

        parser = trt.OnnxParser(network, logger_rt)
        logger.info("Load ONNX file successfully!")
        with open(onnx_file_path, "rb") as onnx_model:
            if not parser.parse(onnx_model.read()):
                logger.info("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))
                exit()
            logger.info("Succeeded parsing .onnx file!")
        
        if self.shape_optim_list is not None:
            for i, input_name in enumerate(self.input_names):
                inputTensor = network.get_input(i)
                if inputTensor.name in self.shape_optim_list:
                    profile.set_shape(inputTensor.name,
                                    self.shape_optim_list[inputTensor.name][0],
                                    self.shape_optim_list[inputTensor.name][1],
                                    self.shape_optim_list[inputTensor.name][2])
                    config.add_optimization_profile(profile)

        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            logger.info("Failed building engine!")
            exit()
        logger.info("Succeeded building engine!")
        with open(trt_file_path, "wb") as f:
            f.write(engineString)
        logger.info("Finish exporting TensorRT model to " + trt_file_path + " !")