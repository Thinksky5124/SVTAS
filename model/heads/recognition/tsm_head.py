'''
Author: Thyssen Wen
Date: 2022-04-29 10:48:58
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-16 21:04:06
Description: TSM action recognition head
FilePath     : /ETESVS/model/heads/tsm_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init

from ...builder import HEADS

@HEADS.register()
class TSMHead(nn.Module):
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
        if sample_rate % 2 != 0:
            raise NotImplementedError

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        # feature [N * num_segs, 1280, 7, 7]

        # feature [N * num_segs, 1280, 1, 1]
        feature = self.avgpool(feature)

        # [N * num_segs, 2048]
        feature = torch.squeeze(feature)

        if self.dropout is not None:
            feature = self.dropout(feature)  # [N, in_channel, num_seg]

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