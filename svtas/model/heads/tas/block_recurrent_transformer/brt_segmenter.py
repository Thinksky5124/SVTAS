'''
Author       : Thyssen Wen
Date         : 2023-02-28 08:47:36
LastEditors  : Thyssen Wen
LastEditTime : 2023-03-02 17:19:27
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/block_recurrent_transformer/brt_segmenter.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from ....builder import HEADS
from ...utils.attention_helper.attention_layer import MultiHeadChunkAttentionLayer, padding_to_multiple_of, LinearChunkAttentionLayer
from .block_recurrent_transformer import RecurrentAttentionBlock

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)
    
class RecurrentHierarchicalAttentionLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 dim_state: int,
                 dim_head: int = 64,
                 state_len: int = 512,
                 num_head: int = 8,
                 dilation: int = 8,
                 causal: bool = False,
                 memory=False,
                 dropout=0.0) -> None:
        super().__init__()
        self.memory = memory
        if memory:
            self.att = RecurrentAttentionBlock(dim, dim_state, dim_head=dim_head,
                                            state_len=state_len, heads=num_head, causal=causal)
            if dropout > 0.0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = None
        else:
            self.att = LinearChunkAttentionLayer(embed_dim=dim, num_heads=num_head, chunck_size=dilation,
                                                 dropout=dropout, position_encoding=True)
            # self.att = MultiHeadChunkAttentionLayer(embed_dim=dim, num_heads=num_head, chunck_size=dilation,
            #                                      dropout=dropout, position_encoding=True)
        self.state = None
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.chunck_size = dilation
    
    def _clear_memory_buffer(self):
        self.state = None

    def gen_causal_mask(self, input_size, device):
        """
        Generates a causal mask of size (input_size, input_size) for attention
        """
        # [T, T]
        l_l_mask = torch.triu(torch.ones(input_size, input_size), diagonal=1) == 1
        l_l_mask = l_l_mask.to(device)
        return l_l_mask
    
    def gen_key_padding_mask(self, masks):
        sum_masks = torch.sum(masks.squeeze(1), dim=-1) == 0.0
        for bs in range(sum_masks.shape[0]):
            if sum_masks[bs]:
                masks[bs] = 1.0
        return (masks == 0.0).squeeze(1)
    
    def forward(self, x, value, masks):
        if self.memory:
            x = torch.transpose(x, 1, 2)

            # chunck
            # padding for groups
            padding = padding_to_multiple_of(x.shape[1], self.chunck_size)
            temporal_size = x.shape[1]
            if padding > 0:
                x = F.pad(x, (0, 0, 0, padding), value = 0.)
            g_size = x.shape[1] // self.chunck_size
            x = rearrange(x, 'b (g n) d -> (b g) n d', n = self.chunck_size)

            # token mix
            x, state = self.att(x, state=self.state)
            self.state = state.detach()
            if self.dropout is not None:
                x = self.dropout(x)

            # segmenter transformer
            x = self.fc1(x)
            x = self.act(x)

            x = rearrange(x, '(b g) n d -> b (g n) d', g = g_size)
            x = torch.transpose(x, 1, 2)[:, :, :temporal_size]
        else:
            x = self.att(x, x, x, masks)
        
        return x * masks[:, 0:1, :]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dim_head=128, state_len=512, num_head=1, stage='encoder', causal=False, memory=False, dropout=0.0):
        super(AttModule, self).__init__()
        self.stage = stage
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = RecurrentHierarchicalAttentionLayer(dim = out_channels,
                                                             dim_state = out_channels,
                                                             dim_head = dim_head,
                                                             state_len = state_len,
                                                             num_head = num_head,
                                                             dilation= dilation,
                                                             causal = causal,
                                                             memory=memory,
                                                             dropout=dropout)
        self.conv_1x1 = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout()
    
    def _clear_memory_buffer(self):
        self.att_layer._clear_memory_buffer()

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        if self.stage == 'encoder':
            out = self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class Encoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, channel_masking_rate,
                 num_head=1, dim_head=128, state_len=512, causal=False, dropout=0.0):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, dim_head=dim_head, state_len=state_len,
                       num_head=num_head, stage='encoder', causal=causal, dropout=dropout, memory=True if i == 4 else False)
                       for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate
    
    def _clear_memory_buffer(self):
        for layer in self.layers:
            layer._clear_memory_buffer()

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
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes,
                 num_head=1, dim_head=128, state_len=512, causal=False, dropout=0.0):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, dim_head=dim_head, state_len=state_len,
                       num_head=num_head, stage='decoder', causal=causal, dropout=dropout)
                       for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def _clear_memory_buffer(self):
        for layer in self.layers:
            layer._clear_memory_buffer()

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

@HEADS.register()
class BRTSegmentationHead(nn.Module):
    """
    LinFormer Head for action segmentation
    """
    def __init__(self,
                 num_head=1,
                 dim_head=128,
                 state_len=512,
                 causal=False,
                 num_decoders=3,
                 encoder_num_layers=10,
                 decoder_num_layers=10,
                 num_f_maps=64,
                 input_dim=2048,
                 num_classes=11,
                 dropout=0.5,
                 att_dropout=0.0,
                 channel_masking_rate=0.5,
                 sample_rate=1,
                 out_feature=False):
        super(BRTSegmentationHead, self).__init__()

        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = Encoder(encoder_num_layers, num_f_maps, input_dim, num_classes, channel_masking_rate,
                               num_head=num_head, dim_head=dim_head, state_len=state_len, causal=causal, dropout=dropout)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(decoder_num_layers, num_f_maps, num_classes, num_classes,
                                                             num_head=num_head, dim_head=dim_head, state_len=state_len, causal=causal, dropout=dropout))
                                                             for s in range(num_decoders)]) # num_decoders
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.encoder._clear_memory_buffer()
        for decoder in self.decoders:
            decoder._clear_memory_buffer()
    
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
    