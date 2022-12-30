'''
Author       : Thyssen Wen
Date         : 2022-06-15 21:14:51
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 21:26:55
Description  : Identity Head
FilePath     : /ETESVS/model/heads/feature_extractor/identity_embedding_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import HEADS

@HEADS.register()
class IdentityEmbeddingHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 sample_rate=4):
        super().__init__()
        self.sample_rate = sample_rate
        if in_channels != out_channels:
            self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.project = None
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
    
    def forward(self, x, masks):
        if self.project is not None:
            x = self.project(x)
        x = x * masks[:, 0:1, ::self.sample_rate]
        return x