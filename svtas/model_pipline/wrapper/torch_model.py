'''
Author       : Thyssen Wen
Date         : 2023-09-21 20:35:44
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-12 15:32:12
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/wrapper/torch_model.py
'''
from typing import Any, Dict
from .base import BaseModel
import torch
from svtas.model_pipline.torch_utils import load_state_dict
import copy
from typing import Iterable, List, Optional, Union

import torch.nn as nn
    
class TorchBaseModel(torch.nn.Module, BaseModel):
    def __init__(self,
                 weight_init_cfg: Union[dict, List[dict], None] = None) -> None:
        super().__init__()
        self.weight_init_cfg = copy.deepcopy(weight_init_cfg)
    
    def _clear_memory_buffer(self):
        pass

    def train(self, val: bool = True):
        self._training = val
        return super().train(val)
    
    def init_weights(self, init_cfg: Dict = {}):
        if self.pretrained is not None:
            state_dict = torch.load(self.pretrained)
            load_state_dict(self, state_dict)
    
    def __repr__(self):
        s = super().__repr__()
        if self.weight_init_cfg:
            s += f'\nweight_init_cfg={self.weight_init_cfg}'
        return s

    def eval(self):
        return super().eval()

    def run_train(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement train function!")
    
    def run_test(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement infer function!")
    
    def forward(self, *args: Any, **kwds: Any) -> Any:
        if self.training:
            return self.run_train(*args, **kwds)
        else:
            return self.run_test(*args, **kwds)

class TorchSequential(TorchBaseModel, nn.Sequential):
    """Sequential module in openmmlab.

    Ensures that all modules in ``Sequential`` have a different initialization
    strategy than the outer model

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        TorchBaseModel.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class TorchModuleList(TorchBaseModel, nn.ModuleList):
    """ModuleList in openmmlab.

    Ensures that all modules in ``ModuleList`` have a different initialization
    strategy than the outer model

    Args:
        modules (iterable, optional): An iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 modules: Optional[Iterable] = None,
                 init_cfg: Optional[dict] = None):
        TorchBaseModel.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class TorchModuleDict(TorchBaseModel, nn.ModuleDict):
    """ModuleDict in openmmlab.

    Ensures that all modules in ``ModuleDict`` have a different initialization
    strategy than the outer model

    Args:
        modules (dict, optional): A mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 modules: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        TorchBaseModel.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)