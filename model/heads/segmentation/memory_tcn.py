'''
Author       : Thyssen Wen
Date         : 2022-06-18 12:11:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-19 10:31:40
Description  : memory tcn
FilePath     : /ETESVS/model/heads/segmentation/memory_tcn.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init

from ...builder import HEADS

@HEADS.register()
class MemoryTCNHead(nn.Module):
    def __init__(self,
                 num_stages,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 sample_rate=1,
                 out_feature=False):
        super(MemoryTCNHead, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.stage1 = MemoryTemporalConvolutionBlock(num_layers, num_f_maps, dim, num_classes, out_feature=out_feature)
        self.stages = nn.ModuleList([copy.deepcopy(MemoryTemporalConvolutionBlock(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         kaiming_init(m)
        #     elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        #         constant_init(m, 1)
        pass

    def _clear_memory_buffer(self):
        self.stage1._clear_memory_buffer()
        for s in self.stages:
            s._clear_memory_buffer()

    def forward(self, x, mask):
        mask = mask[:, :, ::self.sample_rate]
        
        output = self.stage1(x, mask)

        if self.out_feature is True:
            feature, out = output
        else:
            out = output

        outputs = out.unsqueeze(0)
        for s in self.stages:
            if self.out_feature is True:
                out, feature = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            else:
                out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        
        if self.out_feature is True:
            return feature, outputs
        return outputs
        
class MemoryDilationResidualLyaer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(MemoryDilationResidualLyaer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.norm = nn.Dropout()
        self.norm = nn.BatchNorm1d(out_channels)
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
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]

class MemoryTemporalConvolutionBlock(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 out_feature=False):
        super(MemoryTemporalConvolutionBlock, self).__init__()
        self.out_feature = out_feature
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(MemoryDilationResidualLyaer(2**(num_layers-1-i), num_f_maps, num_f_maps)) for i in range(num_layers)])
        # self.layers = nn.ModuleList([copy.deepcopy(MemoryDilationResidualLyaer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
    
    def _clear_memory_buffer(self):
        self.apply(self._clean_buffers)

    @staticmethod
    def _clean_buffers(m):
        if issubclass(type(m), MemoryDilationResidualLyaer):
            m._resert_memory()
    
    def forward(self, x, mask):
        feature_embedding = self.conv_1x1(x)
        feature = feature_embedding
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        if self.out_feature is True:
            return feature_embedding * mask[:, 0:1, :], out
        return out