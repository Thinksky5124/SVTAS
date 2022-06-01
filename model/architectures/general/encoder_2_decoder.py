'''
Author       : Thyssen Wen
Date         : 2022-05-27 15:46:21
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-28 15:29:56
Description  : Text Translation framework
FilePath     : /ETESVS/model/architectures/encoder_2_decoder.py
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
        text = input_data['text']
        
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        # x.shape=[N,T,C,H,W], for most commonly case
        text = torch.reshape(text, [-1] + list(text.shape[2:]))
        # x [N * T, C, H, W]

        return text