'''
Author       : Thyssen Wen
Date         : 2022-12-30 16:17:51
LastEditors  : Thyssen Wen
LastEditTime : 2023-01-04 11:00:02
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/block.py
'''
import torch
import torch.nn as nn
from .token_mixer_layer import *
from .ffn import ResdualDilationConvBlock, ResdualMLPBlock

class ResdualAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dilation,
                 dropout=0.0,
                 causal=False,
                 position_encoding=True,
                 mode='encoder') -> None:
        super().__init__()
        # self.att_layer = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
        #                                             dropout=dropout, causal=causal)
        # self.att_layer = GAUAttentionLayer(embed_dim=embed_dim, dropout=dropout, causal=causal)
        # self.att_layer = MixedChunkAttentionLayer(input_dim=embed_dim, causal=causal,
        #                                             dropout=dropout, group_size=dilation,
        #                                             position_encoding=position_encoding)
        # self.att_layer = PoolFormerMixTokenLayer(pool_size=2**dilation, mode=mode)
        # self.att_layer = MultiHeadChunkAttentionLayer(embed_dim=embed_dim, causal=causal,
        #                                             dropout=dropout, chunck_size=dilation,
        #                                             num_heads=num_heads, position_encoding=position_encoding)
        # self.att_layer = GAUAChunkttentionLayer(embed_dim=embed_dim, causal=causal,
        #                                             dropout=dropout, chunck_size=dilation,
        #                                             num_heads=num_heads, position_encoding=position_encoding)
        # self.att_layer = MHRPRChunkAttentionLayer(embed_dim=embed_dim, causal=causal,
        #                                             dropout=dropout, chunck_size=dilation,
        #                                             num_heads=num_heads)
        # self.att_layer = AttLayer(q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim, r1=2, r2=2, r3=2, bl=dilation, att_type='sliding_att')
        # self.prev_norm = nn.InstanceNorm1d(num_features=embed_dim)
        # self.prev_norm = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        # self.feed_forward = ResdualMLPBlock(embed_dim=embed_dim, dropout=dropout)
        # self.after_norm = nn.InstanceNorm1d(num_features=embed_dim)
        # self.after_norm = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        self.token_mixer_block = PoolFormerBlock(dim=embed_dim, drop=dropout, mode=mode)
    
    def forward(self, x, q, k, v, masks):
        # out = self.att_layer(q, k, v, masks) + x
        # out = self.prev_norm(out)
        # out = self.feed_forward(out, masks)
        # out = self.after_norm(out)
        # return out
        return self.token_mixer_block(x, q, k, v, masks)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dilation=1,
                 dropout=0.0,
                 causal=False,
                 position_encoding=True) -> None:
        super().__init__()
        # self.dilation_conv = ResdualDilationConvBlock(in_channels=embed_dim, out_channels=embed_dim,
        #                                               dilation=2**dilation, dropout=dropout)
        # self.norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.att_block = ResdualAttentionBlock(embed_dim=embed_dim, num_heads=num_heads,
                                                 dropout=dropout, dilation=dilation,
                                                 causal=causal, position_encoding=position_encoding)
    
    def forward(self, x, masks):
        # x = self.dilation_conv(x, masks)
        # x = self.norm(x)
        x = self.att_block(x, x, x, x, masks)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dilation=1,
                 dropout=0.0,
                 causal=False,
                 position_encoding=True) -> None:
        super().__init__()
        # self.dilation_conv = ResdualDilationConvBlock(in_channels=embed_dim,
        #                                               out_channels=embed_dim,
        #                                               dilation=2**dilation,
        #                                               dropout=dropout)
        # self.norm = nn.InstanceNorm1d(num_features=embed_dim)
        # self.att_layer = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
        #                                             dropout=dropout, causal=causal)
        # self.att_layer = PoolFormerMixTokenLayer(mode='decoder')
        # self.norm = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        self.att_block = ResdualAttentionBlock(embed_dim=embed_dim, num_heads=num_heads,
                                                 dropout=dropout, dilation=2**(dilation),
                                                 causal=causal, position_encoding=position_encoding,
                                                 mode='decoder')
        # self.feed_forward = ResdualMLPBlock(embed_dim=embed_dim, dropout=dropout)
        
    
    def forward(self, x, v_x, masks):
        # x = self.dilation_conv(x, masks)
        # x = self.norm(x)
        # v_x_a = self.att_layer(v_x, v_x, v_x, masks)
        # v_x = self.norm(v_x_a + v_x)
        # x = self.norm(x)
        # x = self.att_layer(x, x, v_x, masks) + x
        
        # x = self.feed_forward(x, masks) + x
        x = self.att_block(x, x, x, v_x, masks)
        return x