'''
Author       : Thyssen Wen
Date         : 2023-10-19 21:27:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 21:41:04
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/onnxruntime_model.py
'''
import os
from typing import Any, Dict, Sequence, List

from .base import BaseModel
from svtas.utils import is_onnx_available, AbstractBuildFactory

if is_onnx_available():
    import onnxruntime as ort

@AbstractBuildFactory.register('model')
class ONNXRuntimeModel(BaseModel):
    def __init__(self,
                 input_names: List[str],
                 path_or_bytes: str | bytes,
                 output_names: List[str] = None,
                 sess_options: Sequence | None = None,
                 providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
                 provider_options: Sequence[dict[Any, Any]] | None = None,
                 **kwargs) -> None:
        super().__init__()
        self._training = False
        self.input_names = input_names
        self.output_names = output_names
        self.ort_session = ort.InferenceSession(path_or_bytes, sess_options, providers, provider_options, **kwargs)
    
    def train(self, val: bool = False):
        pass

    def _clear_memory_buffer(self):
        return super()._clear_memory_buffer()
    
    def init_weights(self, init_cfg: Dict = {}):
        return super().init_weights(init_cfg)
    
    def forward(self, input_data: Dict) -> Dict:
        data_dict = {}
        for key, value in input_data.items():
            if key in self.input_names:
                data_dict[key] = value
        
        output_list = self.ort_session.run(self.output_names, data_dict)

        if self.output_names is not None and len(self.output_names) == len(output_list):
            output_dict = {}
            for key, value in zip(self.output_names, output_list):
                output_dict[key] = value
            return output_dict
        else:
            return dict(output = output_list[0])
    
    def __call__(self, input_data: Dict) -> Dict:
        return self.forward(input_data=input_data)