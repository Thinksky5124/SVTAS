'''
Author       : Thyssen Wen
Date         : 2022-05-27 15:46:21
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-04 14:24:59
Description  : Text Translation framework
FilePath     : /ETESVS/model/architectures/general/encoder_2_decoder.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...builder import ARCHITECTURE

from ...builder import build_backbone
from ...builder import build_neck
from ...builder import build_head

@ARCHITECTURE.register()
class Encoder2Decoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 head):
        super().__init__()
        if encoder is not None:
            self.encoder = build_backbone(encoder)
        else:
            self.encoder = None
            
        if decoder is not None:
            self.decoder = build_backbone(decoder)
        else:
            self.decoder = None
        
        if head is not None:
            self.head = build_head(head)
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
            self.head.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
    
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

        return x