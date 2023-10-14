'''
Author       : Thyssen Wen
Date         : 2022-05-27 15:46:21
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 16:00:26
Description  : Text Translation framework
FilePath     : /SVTAS/svtas/model/architectures/general/vae.py
'''
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Dict
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import TorchBaseModel

@AbstractBuildFactory.register('model')
class VariationalAutoEncoder(TorchBaseModel):
    def __init__(self,
                 encoder,
                 decoder,
                 weight_init_cfg = dict(
                    encoder = dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]),
                    decoder = dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]),
                    )):
        super().__init__(weight_init_cfg=weight_init_cfg)
        self.component_list = []
        if encoder is not None:
            self.encoder = AbstractBuildFactory.create_factory('model').create(encoder)
            self.component_list.append('encoder')
        else:
            self.encoder = None
            
        if decoder is not None:
            self.decoder = AbstractBuildFactory.create_factory('model').create(decoder)
            self.component_list.append('decoder')
        else:
            self.decoder = None
        
        self.init_weights(weight_init_cfg)
    
    def init_weights(self, init_cfg: dict = {}):
        for component_name in self.component_list:
            if component_name in init_cfg.keys():
                getattr(self, component_name).init_weights(init_cfg[component_name])
            else:
                getattr(self, component_name).init_weights()
    
    def _clear_memory_buffer(self):
        for component_name in self.component_list:
            getattr(self, component_name)._clear_memory_buffer()
    
    def encode(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        return self.encoder(input_data)

    def decode(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Maps the given latent codes
        onto the output space.
        """
        if self.decoder is not None:
            return self.decoder(input_data)
        else:
            return input_data

    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        dataset space map.
        """
        raise NotImplementedError
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        reparameterize distribution
        """
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input x, returns the reconstructed x
        """
        raise NotImplementedError
    
    def forward(self, input_data):
        if self.encoder is not None:
            encode_output_dict = self.encode(input_data)
        else:
            encode_output_dict = {"output": None}

        if self.decoder is not None:
            decode_output_dict = self.decode(encode_output_dict)
        else:
            decode_output_dict = encode_output_dict
        
        return decode_output_dict