'''
Author       : Thyssen Wen
Date         : 2022-12-30 16:11:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-21 14:40:15
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/token_mixer_layer.py
'''
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.attention_helper.attention_layer import MultiHeadChunkAttentionLayer, MHRPRChunkAttentionLayer

class PoolFormerMixTokenLayer(nn.Module):
    """
    Implementation of pooling for PoolFormer
    implement by pytorch (ref:https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py),
    from paper <MetaFormer is Actually What You Need for Vision> :https://arxiv.org/pdf/2111.11418.pdf

    --pool_size: pooling size
    """
    def __init__(self, pool_size=3, stride=1):
        super().__init__()
        # if mode == 'enocder':
        center_pool_size_list = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        # center_pool_size_list = [3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        # else:
        #     center_pool_size_list = [3, 5, 7, 9, 11, 13, 17, 33, 65, 127, 255, 511]
        center_pool_size = center_pool_size_list[pool_size]
        self.pool = nn.AvgPool1d(
            center_pool_size, stride=stride, padding=center_pool_size//2, count_include_pad=False)

    def forward(self, x, masks):
        x = self.pool(x)
        return x * masks[:, 0:1, :]

class ShfitTokenMixerLayer(nn.Module):
    def __init__(self, shift_div=8) -> None:
        super().__init__()
        self.shift_div = shift_div
    
    @staticmethod
    def partial_shfit(x, shift_div=3, shift_len=1):
        shift_x = torch.zeros_like(x)
        n, c, t = x.size()
        g =  c // shift_div

        chunk_start = random.choice(range(shift_div - 3))
        
        # left shfit
        shift_x[:, g * chunk_start:g * (chunk_start+1), :-shift_len] = x[:, g * chunk_start:g * (chunk_start+1), shift_len:]
        # right shfit
        shift_x[:, g * (chunk_start+1):g * (chunk_start+2), shift_len:] = x[:, g * (chunk_start+1):g * (chunk_start+2), :-shift_len]
        # no shift
        shift_x[:, g * (chunk_start+2):, :] = x[:, g * (chunk_start+2):, ]
        return shift_x
    
    @staticmethod
    def offline_shift(x, shift_div=3, shift_len=1):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, T]
        n, c, t = x.size()

        # [N, num_segments, C]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = torch.permute(x, [0, 2, 1])

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold]
        mid_split = x[:, :, fold:2 * fold]
        right_split = x[:, :, 2 * fold:]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :shift_len, :]
        left_split = left_split[:, shift_len:, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :shift_len, :]
        mid_split = mid_split[:, :-shift_len, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, T, C]
        # restore the original dimension
        return torch.permute(out, [0, 2, 1])

    @staticmethod
    def online_shift(x, shift_div=3, shift_len=1):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, T]
        n, c, t = x.size()

        # [N, num_segments, C]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = torch.permute(x, [0, 2, 1])

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold]
        mid_split = x[:, :, fold:2 * fold]
        right_split = x[:, :, 2 * fold:]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        left_split = torch.roll(left_split, shifts=shift_len, dims=1)

        # shift right on num_segments channel in `mid_split`
        mid_split = torch.roll(mid_split, shifts=shift_len, dims=1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, T, C]
        # restore the original dimension
        return torch.permute(out, [0, 2, 1])

    def forward(self, x, shift_len=1):
        return self.partial_shfit(x=x, shift_div=self.shift_div, shift_len=shift_len)

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, T]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DilationConvBlock(nn.Module):
    def __init__(self, dilation, in_channels, hidden_features, dropout) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, hidden_features, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(hidden_features, in_channels, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x, masks):
        out = self.act(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return out * masks[:, 0:1, :]

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class ShfitTokenFormerEncoderBlock(nn.Module):
    def __init__(self, dim, dilation=3, drop=0., chunck_size=16, position_encoding=True):
        super().__init__()
        self.token_norm = nn.InstanceNorm1d(dim)
        self.mlp_norm = nn.InstanceNorm1d(dim)
        self.token_mixer = ShfitTokenMixerLayer(shift_div=8)
        # self.conv = DilationConvBlock(dilation=2**dilation, in_channels=dim, hidden_features=dim, dropout=drop)
        self.ffn = Mlp(in_features=dim, hidden_features=dim, drop=drop)

        self.attn = MultiHeadChunkAttentionLayer(embed_dim=dim, num_heads=1, chunck_size=chunck_size, position_encoding=position_encoding, dropout=0.2)
        self.shift_len = 2**dilation

    def forward(self, x, masks):
        # out = self.conv(x, masks)
        shfit_x = self.token_mixer(x, shift_len=self.shift_len)
        out = self.token_norm(shfit_x)
        out = x + self.attn(out, out, out, masks)
        out = self.ffn(self.mlp_norm(out)) + out
        return out * masks[:, 0:1, :]

class ShfitTokenFormerDecoderBlock(nn.Module):
    def __init__(self, dim, dilation=3, drop=0., chunck_size=16, position_encoding=True):
        super().__init__()
        self.dilation = dilation
        self.token_norm = nn.InstanceNorm1d(dim)
        self.mlp_norm = nn.InstanceNorm1d(dim)
        self.token_mixer = ShfitTokenMixerLayer(shift_div=8)
        # self.conv = DilationConvBlock(dilation=2**dilation, in_channels=dim, hidden_features=dim, dropout=drop)
        self.ffn = Mlp(in_features=dim, hidden_features=dim, drop=drop)

        self.attn = MultiHeadChunkAttentionLayer(embed_dim=dim, num_heads=1, chunck_size=chunck_size, position_encoding=position_encoding, dropout=0.2)
        self.shift_len = 2**dilation

    def forward(self, x, value, masks):
        shfit_x = self.token_mixer(x, shift_len=self.shift_len)
        out = self.token_norm(shfit_x)
        out = x + self.attn(out, out, value, masks)
        out = self.ffn(self.mlp_norm(out)) + out
        return out * masks[:, 0:1, :]