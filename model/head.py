'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:13
LastEditors: Thyssen Wen
LastEditTime: 2022-04-06 13:50:41
Description: model head
FilePath: /ETESVS/model/head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .mstcn import MemoryStage

class ETESVSHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_layers=4,
                 num_stages=1,
                 clip_seg_num=30,
                 sample_rate=4,
                 sliding_window=60,
                 seg_in_channels=2048,
                 num_f_maps=64):
        super().__init__()
        self.seg_in_channels = seg_in_channels
        self.num_f_maps = num_f_maps
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate
        self.num_layers = num_layers

        self.seg_conv = MemoryStage(num_layers, num_f_maps, seg_in_channels,
                                       num_classes, sliding_window, sample_rate)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                MemoryStage(num_layers, num_f_maps, num_classes,
                                 num_classes, sliding_window, sample_rate)) for s in range(num_stages - 1)
        ])

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.seg_conv._clear_memory_buffer()
        for stage in self.stages:
            stage._clear_memory_buffer()
    
    def forward(self, seg_feature, masks):
        # segmentation branch
        # seg_feature [N, in_channels, temporal_len]
        # Interploate upsample

        out = self.seg_conv(seg_feature, masks[:, :, ::self.sample_rate])
        outputs = out.unsqueeze(0)
        # seg_feature [stage_num, N, num_class, temporal_len]
        for s in self.stages:
            out = s(F.softmax(out, dim=1), masks)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        seg_score = outputs
        return seg_score