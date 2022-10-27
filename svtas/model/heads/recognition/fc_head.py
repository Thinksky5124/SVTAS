'''
Author       : Thyssen Wen
Date         : 2022-05-16 20:58:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-16 21:08:21
Description  : Simple FC feature head
FilePath     : /ETESVS/model/heads/fc_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init

from ...builder import HEADS

@HEADS.register()
class FCHead(nn.Module):
    def __init__(self,
                 num_classes,
                 clip_seg_num=15,
                 sample_rate=4,
                 drop_ratio=0.5,
                 in_channels=2048):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_seg_num = clip_seg_num
        self.drop_ratio = drop_ratio

        self.dropout = nn.Dropout(p=self.drop_ratio)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_init(m)

    def _clear_memory_buffer(self):
        pass
    
    def forward(self, feature, masks):
        # segmentation branch
        # feature [N, in_channels, clip_seg_num]

        if self.dropout is not None:
            feature = self.dropout(feature)  # [N, in_channel, num_seg]
        
        feature = torch.reshape(feature.transpose(1, 2), shape=[-1, self.in_channels])

        # [N * num_segs, C]
        score = self.fc(feature)

        # [N, num_class, num_seg]
        score = torch.reshape(
            score, [-1, self.clip_seg_num, self.num_classes]).permute([0, 2, 1])
        score = score * masks[:, 0:1, ::self.sample_rate]
        # [stage_num, N, C, T]
        score = score.unsqueeze(0)
        score = F.interpolate(
            input=score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        return score