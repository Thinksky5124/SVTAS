'''
Author: Thyssen Wen
Date: 2022-05-02 22:15:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-12 09:47:28
Description: avg pooling 3d to 2d neck
FilePath     : /ETESVS/model/necks/avg_pool_neck.py
'''
import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS

@NECKS.register()
class AvgPoolNeck(nn.Module):
    def __init__(self,
                 num_classes=11,
                 in_channels=1280,
                 clip_seg_num=30,
                 drop_ratio=0.5,
                 need_pool=True,
                 need_pre_cls=False):
        super().__init__()
        self.clip_seg_num = clip_seg_num
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.need_pool = need_pool
        self.need_pre_cls = need_pre_cls
        
        self.backbone_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.need_pre_cls is True:
            self.backbone_dropout = nn.Dropout(p=self.drop_ratio)
            self.backbone_cls_fc = nn.Linear(self.in_channels, num_classes)

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, masks):
        # x.shape = [N * num_segs, in_channels, 7, 7] or [N * num_segs, in_channels]
        # masks.shape = [N, C, T]
        feature = x

        if len(list(feature.shape)) == 2:
            feature = feature.unsqueeze(-1).unsqueeze(-1)

        # backbone branch
        # x.shape = [N * num_segs, in_channels, 1, 1]
        backbone_x = self.backbone_avgpool(feature)

        # [N * num_segs, in_channels]
        backbone_x = torch.squeeze(backbone_x)
        # [N, in_channels, num_segs]
        backbone_feature = torch.reshape(backbone_x, shape=[-1, self.clip_seg_num, backbone_x.shape[-1]]).transpose(1, 2) * masks[:, 0:1, :]

        if self.need_pre_cls is True:
            # get pass feature
            backbone_cls_feature = torch.reshape(backbone_feature.transpose(1, 2), shape=[-1, self.in_channels])

            if self.backbone_dropout is not None:
                backbone_cls_feature = self.backbone_dropout(backbone_cls_feature)  # [N, in_channel, num_seg]

            backbone_score = self.backbone_cls_fc(backbone_cls_feature)  # [N * num_seg, num_class]
            backbone_score = torch.reshape(
                backbone_score, [-1, self.clip_seg_num, backbone_score.shape[1]]).permute([0, 2, 1])  # [N, num_class, num_seg]

            if self.need_pool is True:
                # [N, in_channels, num_segs]
                seg_feature = backbone_feature
            else:
                # [N, in_channels, num_segs, 7, 7]
                feature = torch.reshape(feature, shape=[-1, self.clip_seg_num] + list(feature.shape[-3:])).transpose(1, 2)
                seg_feature = feature * masks[:, 0:1, :].unsqueeze(-1).unsqueeze(-1)

            return seg_feature, backbone_score
        else:
            if self.need_pool is True:
                # [N, in_channels, num_segs]
                seg_feature = backbone_feature
            else:
                # [N, in_channels, num_segs, 7, 7]
                feature = torch.reshape(feature, shape=[-1, self.clip_seg_num] + list(feature.shape[-3:])).transpose(1, 2)
                seg_feature = feature * masks[:, 0:1, :].unsqueeze(-1).unsqueeze(-1)
            
            return seg_feature