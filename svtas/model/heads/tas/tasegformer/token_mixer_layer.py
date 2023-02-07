'''
Author       : Thyssen Wen
Date         : 2022-12-30 16:11:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-01-13 23:20:00
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/token_mixer_layer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        center_pool_size_list = [3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        # else:
        #     center_pool_size_list = [3, 5, 7, 9, 11, 13, 17, 33, 65, 127, 255, 511]
        center_pool_size = center_pool_size_list[pool_size]
        self.pool = nn.AvgPool1d(
            center_pool_size, stride=stride, padding=center_pool_size//2, count_include_pad=False)

    def forward(self, x, masks):
        x = self.pool(x)
        return x * masks[:, 0:1, :]

class PoolSmothLayer(nn.Module):
    def __init__(self, dim, dilation=1, stride=1, scale_init=1e-2):
        pool_size = 2**(dilation + 1) + 1
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=stride, padding=pool_size//2, count_include_pad=False)
        self.scale = nn.Parameter(torch.ones(dim) * scale_init)
        self.norm = nn.InstanceNorm1d(dim)

    def forward(self, x):
        # x = self.norm(self.scale.unsqueeze(-1) * self.pool(x) - x))
        x = self.norm(self.pool(x))
        return x

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

class DilationConv(nn.Module):
    def __init__(self, dilation, in_channels, hidden_features, dropout) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, hidden_features, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(hidden_features, in_channels, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.SiLU()

    def forward(self, x, masks):
        out = self.act(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return out * masks[:, 0:1, :]
        
class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, mode='encoder', pool_size=3, drop=0., stride=1):

        super().__init__()
        self.mode = mode
        assert mode in ['encoder', 'decoder'], f"mode {mode} do not support!"
        self.pool_norm = nn.InstanceNorm1d(dim)
        self.mlp_norm = nn.InstanceNorm1d(dim)
        self.conv_norm = nn.InstanceNorm1d(dim)
        self.conv = DilationConv(dilation=2**pool_size, in_channels=dim, hidden_features=dim, dropout=drop)
        self.token_mixer = PoolFormerMixTokenLayer(pool_size=pool_size, stride=stride)
        self.ffn = Mlp(in_features=dim, hidden_features=dim, drop=drop)
        self.pol_smooth = PoolSmothLayer(dim, dilation=pool_size)

    def forward(self, x, masks):
        out = x + self.conv(self.conv_norm(x), masks)
        if self.mode == 'encoder':
            out = out + self.token_mixer(self.pool_norm(out), masks)
            out = out + self.ffn(self.mlp_norm(out))
        else:
            out = out + self.pol_smooth(out)
        return out * masks[:, 0:1, :]