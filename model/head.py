'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:13
LastEditors: Thyssen Wen
LastEditTime: 2022-04-08 10:39:57
Description: model head
FilePath: /ETESVS/model/head.py
'''
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .mstcn import MemoryStage
from .mstcn import SingleStageModel

class ETESVSHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_layers=4,
                 sample_rate=4,
                 sliding_window=60,
                 seg_in_channels=2048,
                 num_f_maps=64):
        super().__init__()
        self.seg_in_channels = seg_in_channels
        self.num_f_maps = num_f_maps
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.num_layers = num_layers

        self.seg_conv = SingleStageModel(num_layers, num_f_maps, seg_in_channels,
                                       num_classes)
        # self.seg_conv = MemoryStage(num_layers, num_f_maps, seg_in_channels,
        #                                num_classes, sliding_window, sample_rate)

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        # self.seg_conv._clear_memory_buffer()
        pass
    
    def forward(self, seg_feature, masks):
        # segmentation branch
        # seg_feature [N, in_channels, temporal_len]
        # Interploate upsample

        out = self.seg_conv(seg_feature, masks[:, :, ::self.sample_rate])
        # seg_feature [stage_num, N, num_class, temporal_len]
        outputs = out.unsqueeze(0)
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        seg_score = outputs
        return seg_score