'''
Author       : Thyssen Wen
Date         : 2022-12-30 16:17:51
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-30 21:10:03
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/block.py
'''
import torch
import torch.nn as nn
from .attention_layer import *

class ResdualAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dialtion,
                 dropout=0.0,
                 causal=False) -> None:
        super().__init__()
        # self.att_layer = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
        #                                             dropout=dropout, causal=causal)
        # self.att_layer = MixedChunkAttentionLayer(input_dim=embed_dim, causal=causal,
        #                                             dropout=dropout, group_size=dialtion)
        self.att_layer = MultiHeadChunkAttentionLayer(embed_dim=embed_dim, causal=causal,
                                                    dropout=dropout, chunck_size=dialtion,
                                                    num_heads=num_heads)
        # self.att_layer = MHRPRChunkAttentionLayer(embed_dim=embed_dim, causal=causal,
        #                                             dropout=dropout, chunck_size=dialtion,
        #                                             num_heads=num_heads)
        # self.prev_norm = nn.InstanceNorm1d(num_features=embed_dim)
        # self.feed_forward = FeedForwardNetwork(embed_dim=embed_dim, dropout=dropout)
        self.after_norm = nn.InstanceNorm1d(num_features=embed_dim)
    
    def forward(self, x, q, k, v, masks):
        out = self.att_layer(q, k, v, masks) + x
        # out = self.prev_norm(out)
        # out = self.feed_forward(out) + out
        out = self.after_norm(out)
        return out
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dilation=1,
                 dropout=0.0,
                 causal=False) -> None:
        super().__init__()
        # self.dilation_conv = ResdualDilationConvBlock(in_channels=embed_dim, out_channels=embed_dim,
        #                                               dilation=dilation, dropout=dropout)
        self.att_block = ResdualAttentionBlock(embed_dim=embed_dim, num_heads=num_heads,
                                                 dropout=dropout, dialtion=dilation,
                                                 causal=causal)
    
    def forward(self, x, masks):
        # x = self.dilation_conv(x, masks)
        x = self.att_block(x, x, x, x, masks)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dilation=1,
                 dropout=0.0,
                 causal=False) -> None:
        super().__init__()
        # self.dilation_conv = ResdualDilationConvBlock(in_channels=embed_dim,
        #                                               out_channels=embed_dim,
        #                                               dilation=dilation,
        #                                               dropout=dropout)
        # self.att_layer = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
        #                                             dropout=dropout, causal=causal)
        # self.att_layer = MixedChunkAttentionLayer(input_dim=embed_dim, causal=causal,
        #                                             dropout=dropout, group_size=dilation)
        # self.norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.att_block = ResdualAttentionBlock(embed_dim=embed_dim, num_heads=num_heads,
                                                 dropout=dropout, dialtion=dilation,
                                                 causal=causal)
    
    def forward(self, x, v_x, masks):
        # x = self.dilation_conv(x, masks)
        # v_x_a = self.att_layer(v_x, v_x, v_x, masks)
        # v_x = self.norm(v_x_a + v_x)
        x = self.att_block(v_x, x, x, v_x, masks)
        return x