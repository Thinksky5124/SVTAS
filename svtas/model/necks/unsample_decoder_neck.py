'''
Author       : Thyssen Wen
Date         : 2022-11-19 14:41:52
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 13:15:37
Description  : file content
FilePath     : /SVTAS/svtas/model/necks/unsample_decoder_neck.py
'''
import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from ..builder import NECKS

class Adaptive3DTo1DPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        # x [N C T H W]
        _, c, t, _, _ = x.shape
        return F.adaptive_avg_pool3d(x, [t, 1, 1]).squeeze(-1).squeeze(-1)

class Up1DConv(nn.Module):
    def __init__(self, in_channel, reduce_factor=2) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2)
        self.conv = nn.Sequential(
                nn.Conv1d(in_channel, in_channel // reduce_factor, 3, padding=1, dilation=1),
                nn.ReLU(),
                # nn.Conv1d(in_channel // reduce_factor, in_channel // reduce_factor, 3, padding=2, dilation=2),
                # nn.ReLU(),
                nn.BatchNorm1d(in_channel // 2))
        self.pool = Adaptive3DTo1DPooling()
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.pool(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

@NECKS.register()
class UnsampleDecoderNeck(nn.Module):
    def __init__(self,
                 in_channels_list=[3072, 2048, 1280],
                 reduce_factor=2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Up1DConv(in_channel=in_channel, reduce_factor=reduce_factor)
            for in_channel in in_channels_list
        ])
        self.pool = Adaptive3DTo1DPooling()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, masks):
        x_out = self.pool(x[len(x) - 1])
        for i, layer in zip(range(len(x) - 1), self.layers):
            x_out = layer(x_out, x[len(x) - 2 - i])
        return x_out * masks