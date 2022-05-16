'''
Author       : Thyssen Wen
Date         : 2022-05-12 15:25:34
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-16 21:03:38
Description  : Timesformer Head ref:https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/heads/timesformer_head.py
FilePath     : /ETESVS/model/heads/timesformer_head.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import trunc_normal_init

from ..builder import HEADS

@HEADS.register()
class TimeSformerHead(nn.Module):
    """Classification head for TimeSformer.
    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 clip_seg_num=15,
                 sample_rate=4,
                 init_std=0.02,
                 in_channels=1024):
        super().__init__()
        self.init_std = init_std
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_seg_num = clip_seg_num
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, masks):
        # [N * num_segs, in_channels]
        score = self.fc_cls(x)
        # [N * num_segs, num_classes]

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