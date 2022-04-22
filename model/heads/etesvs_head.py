'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:13
LastEditors: Thyssen Wen
LastEditTime: 2022-04-20 13:38:48
Description: model head
FilePath: /ETESVS/model/heads/etesvs_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
import copy
from .mstcn import SingleStageModel
from .memory_tcn import MemoryCausalConvolution

from ..builder import HEADS

@HEADS.register()
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
        if sample_rate % 2 != 0:
            raise NotImplementedError

        self.seg_conv = SingleStageModel(num_layers, num_f_maps, seg_in_channels, num_classes)
        # self.spuer_conv_stages = nn.ModuleList([copy.deepcopy(SuperSampleSingleStageModel(num_classes, num_f_maps)) for s in range(sample_rate // 2)])
        # self.seg_conv = MemoryCausalConvolution(num_layers, num_f_maps, seg_in_channels, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_init(m)

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
        # for i in range(len(self.spuer_conv_stages)):
        #     out = self.spuer_conv_stages[i](F.softmax(out, dim=1) * masks[:, 0:1, ::2 ** (self.sample_rate //2 - i)],
        #                                     masks[:, 0:1, ::2 ** (max(self.sample_rate //2 - i - 1, 0))])
        #     out_score = out.unsqueeze(0)
        #     out_score = F.interpolate(
        #         input=out_score,
        #         scale_factor=[1, 2 ** (max(self.sample_rate //2 - i - 1, 0))],
        #         mode="nearest")
        #     outputs = torch.cat((outputs, out_score), dim=0)
        seg_score = outputs
        return seg_score