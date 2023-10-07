'''
Author       : Thyssen Wen
Date         : 2023-09-21 20:35:44
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 14:36:09
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/torch_model.py
'''
import abc
from typing import Any, Dict
from .base import BaseModel
import torch
from mmengine.runner import load_state_dict, load_checkpoint

class TorchModel(torch.nn.Module, BaseModel):
    def __init__(self) -> None:
        super().__init__()
    
    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, init_cfg: Dict = {}):
        if self.pretrained is not None:
            state_dict = torch.load(self.pretrained)
            load_state_dict(self, state_dict)

    def eval(self):
        return super().eval()

    def run_train(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement train function!")
    
    def run_infer(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement infer function!")
    
    def forward(self, *args: Any, **kwds: Any) -> Any:
        if self.training:
            return self.run_train(*args, **kwds)
        else:
            return self.run_infer(*args, **kwds)