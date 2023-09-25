'''
Author       : Thyssen Wen
Date         : 2023-09-21 20:35:44
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 13:24:03
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/torch_model.py
'''
import abc
from typing import Any
from .base import BaseWapper
import torch

class TorchModel(BaseWapper, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self):
        pass

    def train(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement train function!")
    
    def infer(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement infer function!")
    
    def forward(self, *args: Any, **kwds: Any) -> Any:
        if self.training:
            return self.train(*args, **kwds)
        else:
            return self.infer(*args, **kwds)