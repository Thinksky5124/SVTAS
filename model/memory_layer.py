'''
Author: Thyssen Wen
Date: 2022-04-13 14:01:12
LastEditors: Thyssen Wen
LastEditTime: 2022-04-13 16:10:55
Description: file content
FilePath: /ETESVS/model/memory_layer.py
'''
import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from .resnet import BasicBlock
from mmcv.cnn import ConvModule

class RNNConvModule(nn.Module):
    def __init__(self,
                 in_channels=2048,
                 out_channels=2048,
                 hidden_channels=512,
                 memory_len=30,
                 conv_cfg = dict(type='Conv1d'),
                 norm_cfg = dict(type='BN1d', requires_grad=True)):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.memory_len = memory_len
        self.reduce_conv = BasicBlock(self.in_channel,
                                     self.hidden_channels,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=None,
                                     downsample=ConvModule(
                                        self.in_channel,
                                        self.hidden_channels,
                                        kernel_size=1,
                                        conv_cfg=conv_cfg,
                                        act_cfg=None))
        self.memory_conv = ConvModule(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.x_conv = ConvModule(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.tanh = nn.Tanh()
        self.conv_out = ConvModule(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1,
            act_cfg=dict(type='ReLU'),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.softmax = nn.Softmax(dim=-1)
        self.memory = None
        self.dropout = nn.Dropout()

    def _reset_memory(self):
        self.memory = None

    def forward(self, x, masks=None):
        # x.shape[N, hidden_channels, T]
        reduce_x = self.reduce_conv(x)
        if self.memory is None:
            x_shape = list(reduce_x.shape)
            x_shape[1] = self.hidden_channels
            x_shape[-1] = self.memory_len
            self.memory = torch.zeros(x_shape, dtype=x.dtype, device=x.device)
        hidden_out = self.dropout(self.tanh(self.x_conv(reduce_x) + self.memory_conv(self.memory)))
        out = self.conv_out(hidden_out)
        if masks is not None:
            out = out * masks[:, 0:1, :]

        self.memory = hidden_out.detach().clone()

        return out

class SuperSampleSingleStageModel(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, hidden_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.transpose_conv = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.dialtion_conv_1 = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.dialtion_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout()
    
    def forward(self, x, masks):
        up_x = self.upsample(x)
        x = self.conv_1x1(x)
        x = self.transpose_conv(x)
        x = self.dialtion_conv_1(x)
        x = self.dialtion_conv_2(x)
        x = self.dropout(x)
        return (x + up_x) * masks[:, 0:1, :]
        

class SlidingDilationResidualLyaer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(SlidingDilationResidualLyaer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.dilation = dilation
        self.memory = None
    
    def _resert_memory(self):
        self.memory = None
    
    def _memory(self, x):
        self.memory = x[:, :, -(self.dilation * 2):].detach().clone()

    def overlap_same_padding(self, x):
        if self.memory is None:
            self.memory = torch.zeros([x.shape[0], x.shape[1], self.dilation * 2]).to(x.device)
        
        # overlap
        x = torch.cat([self.memory, x], dim=2)
        self._memory(x)
        return x
    
    def forward(self, x, mask):
        # padding
        pad_x = self.overlap_same_padding(x)

        out = F.relu(self.conv_dilated(pad_x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class MemoryStage(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(MemoryStage, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        # self.layers = nn.ModuleList([copy.deepcopy(SlidingDilationResidualLyaer(2**(num_layers-1-i), num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.layers = nn.ModuleList([copy.deepcopy(SlidingDilationResidualLyaer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
    
    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out
    
    def _clear_memory_buffer(self):
        self.apply(self._clean_buffers)

    @staticmethod
    def _clean_buffers(m):
        if issubclass(type(m), SlidingDilationResidualLyaer):
            m._resert_memory()