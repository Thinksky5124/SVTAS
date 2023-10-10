'''
Author       : Thyssen Wen
Date         : 2022-05-27 15:46:21
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 14:26:29
Description  : Text Translation framework
FilePath     : /SVTAS/svtas/model/architectures/general/autoencoder.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import TorchModel

@AbstractBuildFactory.register('architecture')
class AutoEncoder(TorchModel):
    def __init__(self,
                 encoder,
                 decoder,
                 head):
        super().__init__()
        if encoder is not None:
            self.encoder = AbstractBuildFactory.create_factory('model').create(encoder)
        else:
            self.encoder = None
            
        if decoder is not None:
            self.decoder = AbstractBuildFactory.create_factory('model').create(decoder)
        else:
            self.decoder = None
        
        if head is not None:
            self.head = AbstractBuildFactory.create_factory('model').create(head)
            self.sample_rate = head.sample_rate
        else:
            self.head = None
    
        self.init_weights()
    
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