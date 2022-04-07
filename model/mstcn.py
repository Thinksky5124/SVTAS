'''
Author: Thyssen Wen
Date: 2022-03-25 20:31:27
LastEditors: Thyssen Wen
LastEditTime: 2022-04-07 10:56:10
Description: ms-tcn script ref: https://github.com/yabufarha/ms-tcn
FilePath: /ETESVS/model/mstcn.py
'''
from turtle import forward
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class SlidingDilationResidualLyaer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, sliding_window, sample_rate):
        super(SlidingDilationResidualLyaer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.dilation = dilation
        self.sliding = sliding_window // sample_rate
        self.memory = None
    
    def _resert_memory(self):
        self.memory = None
    
    def _memory(self, x):
        over_x = x[:, :, :(self.sliding + self.dilation * 2)]
        self.memory = over_x[:, :, -(self.dilation * 2):].detach().clone()

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
    def __init__(self, num_layers, num_f_maps, dim, num_classes, sliding_window, sample_rate):
        super(MemoryStage, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(SlidingDilationResidualLyaer(2**(num_layers-1-i), num_f_maps, num_f_maps, sliding_window, sample_rate)) for i in range(num_layers)])
        # self.layers = nn.ModuleList([copy.deepcopy(SlidingDilationResidualLyaer(2 ** i, num_f_maps, num_f_maps, sliding_window, sample_rate)) for i in range(num_layers)])
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