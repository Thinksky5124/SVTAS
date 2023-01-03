'''
Author       : Thyssen Wen
Date         : 2022-12-22 20:15:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-01-03 17:36:46
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/tasegformer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerDecoderBlock, TransformerEncoderBlock
import numpy as np
import copy
from ....builder import HEADS

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_heads,
                 num_layers,
                 num_classes,
                 dropout=0.0,
                 position_encoding=True):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, embed_dim, 1)
        self.position_encoding = position_encoding
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, dilation=i)
                for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(embed_dim, num_classes, 1)

    def forward(self, x, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return None, feature


class Decoder(nn.Module):
    def __init__(self,
                 value_channels,
                 embed_dim,
                 num_heads,
                 num_layers,
                 num_classes,
                 dropout=0.0,
                 position_encoding=True):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(value_channels, embed_dim, 1)
        self.position_encoding = position_encoding
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, dilation=i)
                for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(embed_dim, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
        
@HEADS.register()
class TASegFormer(nn.Module):
    def __init__(self,
                 in_channels,
                 num_decoders=3,
                 decoder_num_layers=10,
                 encoder_num_layers=10,
                 num_classes=11,
                 embed_dim=64,
                 num_heads=8,
                 dropout=0.5,
                 sample_rate=1,
                 out_feature=False,
                 position_encoding=True):
        super(TASegFormer, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = Encoder(in_channels=in_channels,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_layers=encoder_num_layers,
                               num_classes=num_classes,
                               dropout=dropout,
                               position_encoding=position_encoding)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(in_channels=num_classes,
                                                             embed_dim=embed_dim,
                                                             num_heads=num_heads,
                                                             num_layers=decoder_num_layers,
                                                             num_classes=num_classes,
                                                             dropout=dropout,
                                                             position_encoding=position_encoding))
                                                             for s in range(num_decoders)]) # num_decoders
        
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
                
    def forward(self, x, mask):
        # x.shape [N C T]
        # mask.shape [N C T]
        
        out, feature = self.encoder(x, mask[:, 0:1, ::self.sample_rate])
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, ::self.sample_rate], feature * mask[:, 0:1, ::self.sample_rate], mask[:, 0:1, ::self.sample_rate])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        
        if self.out_feature is True:
            return feature, outputs
        return outputs