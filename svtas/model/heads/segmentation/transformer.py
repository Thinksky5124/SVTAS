'''
Author       : Thyssen Wen
Date         : 2022-12-22 20:15:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-23 17:08:27
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/segmentation/transformer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import math
from ...builder import HEADS

class FeedNetwork(nn.Module):
    def __init__(self,
                 embed_dim) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU())
        
    def forward(self, x):
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,) -> None:
        super().__init__()
        self.feed_forward = FeedNetwork()
        self.att_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.prev_norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.feed_forward = FeedNetwork()
        self.after_norm = nn.InstanceNorm1d(num_features=embed_dim)
    
    def forward(self, q, k, v):
        x, att_map = self.att_layer(q, k, v)
        x = self.prev_norm(x)
        x = self.feed_forward(x)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,) -> None:
        super().__init__()
        self.att_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.prev_norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.feed_forward = FeedNetwork()
        self.after_norm = nn.InstanceNorm1d(num_features=embed_dim)
    
    def forward(self, x, mask):
        x = self.att_layer(x)
        pass

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,) -> None:
        super().__init__()
        self.att_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.prev_norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.feed_forward = FeedNetwork()
        self.after_norm = nn.InstanceNorm1d(num_features=embed_dim)
    
    def forward(self, q, k, v, mask):
        x = self.att_layer(x)
        pass

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [TransformerBlock()
                for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        # feature = self.position_en(feature)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
        
@HEADS.register()
class ASFormer(nn.Module):
    def __init__(self,
                 num_decoders=3,
                 num_layers=10,
                 r1=2,
                 r2=2,
                 num_f_maps=64,
                 input_dim=2048,
                 num_classes=11,
                 channel_masking_rate=0.5,
                 sample_rate=1,
                 out_feature=False):
        super(ASFormer, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
        
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
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, ::self.sample_rate], feature* mask[:, 0:1, ::self.sample_rate], mask[:, 0:1, ::self.sample_rate])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        if self.out_feature is True:
            return feature, outputs
        return outputs