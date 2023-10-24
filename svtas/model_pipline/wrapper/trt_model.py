'''
Author       : Thyssen Wen
Date         : 2023-10-22 15:15:23
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 19:34:24
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/trt_model.py
'''
import numpy as np
from typing import Any, Dict, Sequence, List
from .base import BaseModel
from svtas.utils import AbstractBuildFactory, is_tensorrt_available

if is_tensorrt_available():
    import tensorrt as trt
    from cuda import cudart
    from svtas.utils.logger import TensorRTLogger

@AbstractBuildFactory.register('model')
class TensorRTModel(BaseModel):
    def __init__(self,
                 model_path: str,
                 input_names: List[str],
                 output_names: List[str] = None) -> None:
        super().__init__()
        self._training = False
        self.input_names = input_names
        self.output_names = output_names
        trt_logger = TensorRTLogger()

        with open(model_path, 'rb') as f:
                serialized_engine = f.read()
                runtime = trt.Runtime(trt_logger)
                self.trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
                
        cudart.cudaDeviceSynchronize()
        self.num_io = self.trt_engine.num_io_tensors
        self.io_tensor_names = [self.trt_engine.get_tensor_name(i) for i in range(self.num_io)]
        self.num_inputs = [self.trt_engine.get_tensor_mode(self.io_tensor_names[i]) for i in range(self.num_io)].count(trt.TensorIOMode.INPUT)

    def train(self, val: bool = False):
        pass

    def _clear_memory_buffer(self):
        return super()._clear_memory_buffer()
    
    def init_weights(self, init_cfg: Dict = {}):
        return super().init_weights(init_cfg)
    
    def forward(self, input_data: Dict[str, np.array]) -> Dict:
        # get input data_dict
        data_dict = {}
        for key, value in input_data.items():
            if key in self.input_names:
                data_dict[key] = value

        data_device_dict = {}
        for key, value in data_dict.items():
            data_device_dict[key] = cudart.cudaMalloc(value.nbytes)[1]
            cudart.cudaMemcpy(data_device_dict[key],
                              data_dict[key].ctypes.data,
                              data_dict[key].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # set input shape
        context = self.trt_engine.create_execution_context()
        for key, value in data_dict.items():
            context.set_input_shape(key, list(value.shape))
        
        # set output shape
        output_dict = {}
        if self.output_names is None:
            for i in range(self.num_inputs, self.num_io):
                output_dict[self.io_tensor_names[i]] = np.empty(context.get_tensor_shape(self.io_tensor_names[i]),
                                                                dtype=trt.nptype(self.trt_engine.get_tensor_dtype(self.io_tensor_names[i])))
        else:
            for name in self.output_names:
                output_dict[name] = np.empty(context.get_tensor_shape(name),
                                             dtype=trt.nptype(self.trt_engine.get_tensor_dtype(name)))

        output_device_dict = {}
        for key, value in output_dict.items():
            output_device_dict[key] = cudart.cudaMalloc(value.nbytes)[1]
            cudart.cudaMemcpy(output_device_dict[key],
                              output_dict[key].ctypes.data,
                              output_dict[key].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # exec infer graph
        for key, value in data_device_dict.items():
            context.set_tensor_address(key, int(value))
        for key, value in output_device_dict.items():
            context.set_tensor_address(key, int(value))

        context.execute_async_v3(0)

        # extract output tensor
        for key, value in output_dict.items():
            cudart.cudaMemcpy(output_dict[key].ctypes.data,
                              output_device_dict[key],
                              output_dict[key].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # output memory free
        for key, value in output_device_dict.items():
            cudart.cudaFree(value)
        # input memory free
        for key, value in data_device_dict.items():
            cudart.cudaFree(value)
        return output_dict

    def __call__(self, input_data: Dict) -> Dict:
        return self.forward(input_data=input_data)