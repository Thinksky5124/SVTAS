'''
Author       : Thyssen Wen
Date         : 2023-09-25 13:34:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-17 10:37:43
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/general/serious.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import TorchBaseModel

@AbstractBuildFactory.register('model')
class SeriousModel(TorchBaseModel):
    def __init__(self,
                 weight_init_cfg = {},
                 **kwargs) -> None:
        super().__init__(weight_init_cfg)
        self.component_list = []
        for name, cfg in kwargs.items():
            if cfg is not None:
                setattr(self, name, AbstractBuildFactory.create_factory('model').create(cfg))
                self.component_list.append(name)
            else:
                setattr(self, name, None)
    
    def init_weights(self, init_cfg: dict = {}):
        if len(init_cfg) <= 0 or init_cfg is None:
            init_cfg = self.weight_init_cfg
        for component_name in self.component_list:
            if component_name in init_cfg.keys():
                getattr(self, component_name).init_weights(init_cfg[component_name])
            else:
                getattr(self, component_name).init_weights()
    
    def _clear_memory_buffer(self):
        for component_name in self.component_list:
            getattr(self, component_name)._clear_memory_buffer()
