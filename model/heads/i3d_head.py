'''
Author: Thyssen Wen
Date: 2022-04-30 14:27:47
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-17 20:24:20
Description: I3D head ref:https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/heads/i3d_head.py
FilePath     : /ETESVS/model/heads/i3d_head.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS


@HEADS.register()
class I3DHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 clip_seg_num=15,
                 sample_rate=4,
                 drop_ratio=0.5,
                 init_std=0.01):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.drop_ratio = drop_ratio
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate
        self.init_std = init_std
        if self.drop_ratio != 0:
            self.dropout = nn.Dropout(p=self.drop_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
    
    def _clear_memory_buffer(self):
        pass

    def forward(self, feature, masks):
        """Defines the computation performed at every call.

        Args:
            feature (torch.Tensor): The input data.
            masks   (torch.Tensor): The mask.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, clip_seg_num, 7, 7] -> [N * clip_seg_num, in_channels, 7, 7]
        feature = torch.reshape(feature.transpose(1, 2), [-1, self.in_channels] + list(feature.shape[-2:]))
        # [N, in_channels, clip_seg_num, 7, 7]
        if self.avg_pool is not None:
            feature = self.avg_pool(feature)
        # [N, in_channels, clip_seg_num, 1, 1]
        if self.dropout is not None:
            feature = self.dropout(feature)
        # [N, clip_seg_num, in_channels]
        feature = torch.transpose(feature, 1, 2).squeeze(-1).squeeze(-1)
        # [N * clip_seg_num, in_channels]
        feature = torch.reshape(feature, [-1, self.in_channels])
        # [N * clip_seg_num, num_classes]
        score = self.fc_cls(feature)
        # [N, num_class, clip_seg_num]
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