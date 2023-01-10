'''
Author       : Thyssen Wen
Date         : 2022-12-22 20:15:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-01-09 16:22:43
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/tasegformer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .token_mixer_layer import *
import numpy as np
import copy
from ....builder import HEADS

class DilationConvBlock(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dropout) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask):
        out = self.act(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class ResdualTokenMixerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 dilation,
                 dropout=0.0,
                 mode='encoder') -> None:
        super().__init__()
        # self.dilation_conv = DilationConvBlock(2**(dilation), in_channels=embed_dim, out_channels=embed_dim, dropout=dropout)
        self.token_mixer_blocks = nn.ModuleList([PoolFormerBlock(dim=embed_dim, drop=dropout, pool_size=(2**(dilation+1)+1), stride=1, mode=mode)
                                                 for i in range(1)])
        
    def forward(self, x, masks=None):
        # x = self.dilation_conv(x, masks)
        for token_mixer_block in self.token_mixer_blocks:
            x = token_mixer_block(x, masks)
        return x

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_layers,
                 num_classes,
                 dropout=0.0):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, embed_dim, 1)
        self.layers = nn.ModuleList(
            [ResdualTokenMixerBlock(embed_dim=embed_dim,
                                   dilation=i,
                                   dropout=dropout,
                                   mode='encoder')
                for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(embed_dim, num_classes, 1)

    def forward(self, x, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        
        out = self.conv_out(feature)

        return out, feature


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_layers,
                 num_classes,
                 dropout=0.0):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, embed_dim, 1)
        self.layers = nn.ModuleList(
            [ResdualTokenMixerBlock(embed_dim=embed_dim,
                                   dilation=i,
                                   dropout=dropout,
                                   mode='decoder')
                for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(embed_dim, num_classes, 1)

    def forward(self, x, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
        
@HEADS.register()
class TASegFormer(nn.Module):
    def __init__(self,
                 in_channels,
                 num_decoders=3,
                 decoder_num_layers=10,
                 encoder_num_layers=10,
                 input_dropout_rate=0.5,
                 num_classes=11,
                 embed_dim=64,
                 dropout=0.5,
                 sample_rate=1,
                 out_feature=False):
        super(TASegFormer, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = Encoder(in_channels=in_channels,
                               embed_dim=embed_dim,
                               num_layers=encoder_num_layers,
                               num_classes=num_classes,
                               dropout=dropout)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(in_channels=num_classes,
                                                             embed_dim=embed_dim,
                                                             num_layers=decoder_num_layers,
                                                             num_classes=num_classes,
                                                             dropout=dropout))
                                                             for s in range(num_decoders)]) # num_decoders
        self.input_dropout_rate = input_dropout_rate
        assert 0.0 <= self.input_dropout_rate < 1.0, f"input_dropout_rate must between 0.0~1.0, now is {input_dropout_rate}!"
        if self.input_dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(p=input_dropout_rate)
        else:
            self.dropout = None

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
                
    def forward(self, x, mask):
        # x.shape [N C T]
        # mask.shape [N C T]
        
        if self.dropout is not None:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)
            
        out, feature = self.encoder(x, mask[:, 0:1, ::self.sample_rate])
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, d_feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, ::self.sample_rate], mask[:, 0:1, ::self.sample_rate])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")[:, :, :, :mask.shape[-1]]
        
        if self.out_feature is True:
            return feature, outputs
        return outputs