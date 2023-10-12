'''
Author       : Thyssen Wen
Date         : 2023-09-21 15:56:43
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-12 10:17:59
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/general/diffusion.py
'''
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VariationalAutoEncoders
from ...scheduler import BaseDiffusionScheduler
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import TorchBaseModel

@AbstractBuildFactory.register('model')
class DiffusionModel(TorchBaseModel):
    vae: VariationalAutoEncoders
    scheduler: BaseDiffusionScheduler

    def __init__(self,
                 vae: Dict,
                 unet: Dict,
                 scheduler: Dict,
                 weight_init_cfg: dict | List[dict] | None = None) -> None:
        super().__init__(weight_init_cfg)
    
    def init_weights(self):
        if self.encoder is not None:
            self.encoder.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        if self.decoder is not None:
            self.decoder.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        if self.head is not None:
            self.head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.encoder is not None:
            self.encoder._clear_memory_buffer()
        if self.decoder is not None:
            self.decoder._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()
    
    def forward(self, input_data):
        masks = input_data['masks']
        x = input_data['x']
        
        if self.encoder is not None:
            x = self.encoder(x, masks)
        else:
            x = x

        if self.decoder is not None:
            x = self.decoder(x, masks)
        else:
            x = x
        
        if self.head is not None:
            x = self.head(x, masks)
        else:
            x = x
        
        return {"output":x}