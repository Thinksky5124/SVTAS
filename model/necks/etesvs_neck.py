'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-25 23:12:06
Description: model neck
FilePath: /ETESVS/model/necks/etesvs_neck.py
'''
import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from .memory_layer import ConvLSTMResidualLayer

from ..builder import NECKS

@NECKS.register()
class ETESVSNeck(nn.Module):
    def __init__(self,
                 num_classes=11,
                 num_layers=1,
                 cls_in_channel=1280,
                 cls_hidden_channel=1280,
                 seg_in_channel=320,
                 seg_hidden_channel=320,
                 clip_seg_num=30,
                 drop_ratio=0.5,
                 bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.clip_seg_num = clip_seg_num
        self.cls_in_channel = cls_in_channel
        self.seg_in_channel = seg_in_channel
        self.cls_hidden_channel = cls_hidden_channel
        self.seg_hidden_channel = seg_hidden_channel
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.bidirectional = bidirectional
        
        self.backbone_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.neck_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.rnn_conv = ConvLSTMResidualLayer(self.seg_in_channel, self.seg_hidden_channel, self.num_classes, self.num_layers,
        #                         dropout=self.drop_ratio, bidirectional=self.bidirectional)
        # self.neck_cls = nn.Conv1d(self.seg_hidden_channel, self.num_classes, 1)

        self.backbone_dropout = nn.Dropout(p=self.drop_ratio)
        self.neck_dropout = nn.Dropout(p=self.drop_ratio)
        # self.neck_cls_dropout = nn.Dropout(p=self.drop_ratio)
        
        self.backbone_cls_conv = nn.Conv1d(self.cls_hidden_channel, self.seg_hidden_channel, 1)
        self.backbone_cls_fc_2 = nn.Linear(self.seg_hidden_channel, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def _clear_memory_buffer(self):
        # self.rnn_conv._reset_memory()
        pass

    def forward(self, x, masks):
        # x.shape = [N * num_segs, 320, 7, 7] [N * num_segs, 1280, 7, 7]
        seg_feature, reg_feature = x
        # backbone branch
        # x.shape = [N * num_segs, 2048, 1, 1]
        backbone_x = self.backbone_avgpool(reg_feature)

        # [N * num_segs, 2048]
        backbone_x = torch.squeeze(backbone_x)
        # [N, 2048, num_segs]
        backbone_feature = torch.reshape(backbone_x, shape=[-1, self.clip_seg_num, backbone_x.shape[-1]]).transpose(1, 2)
        backbone_feature = self.backbone_cls_conv(backbone_feature)  # [N, hidden_channel, num_seg]

        # get pass feature
        backbone_cls_feature = torch.reshape(backbone_feature.transpose(1, 2), shape=[-1, self.seg_hidden_channel])

        if self.backbone_dropout is not None:
            backbone_cls_feature = self.backbone_dropout(backbone_cls_feature)  # [N, in_channel, num_seg]

        backbone_score = self.backbone_cls_fc_2(backbone_cls_feature)  # [N * num_seg, num_class]
        backbone_score = torch.reshape(
            backbone_score, [-1, self.clip_seg_num, backbone_score.shape[1]]).permute([0, 2, 1])  # [N, num_class, num_seg]

        # segmentation feature branch
        # masks.shape = [N, T]
        # seg_feature = torch.reshape(seg_feature, shape=[-1, self.clip_seg_num] + list(seg_feature.shape[-3:]))

        # # memory branch
        # # [N, num_segs, 1280, 7, 7]
        # spatio_temporal_feature = self.rnn_conv(seg_feature, masks)

        # spatio_temporal_feature = torch.reshape(spatio_temporal_feature, shape=[-1] + list(seg_feature.shape[-3:]))
        # # seg_feature_pool.shape = [N * num_segs, 1280, 1, 1]
        # spatio_temporal_feature_pool = self.neck_avgpool(spatio_temporal_feature)

        # # seg_feature_pool.shape = [N, num_segs, 1280, 1, 1]
        # spatio_temporal_feature_pool = torch.reshape(spatio_temporal_feature_pool, shape=[-1, self.clip_seg_num] + list(spatio_temporal_feature_pool.shape[-3:]))

        # # segmentation feature branch
        # # [N, num_segs, 2048]
        # spatio_temporal_feature_pool = spatio_temporal_feature_pool.squeeze(-1).squeeze(-1)
        # # [N, 2048, num_segs]
        # spatio_temporal_feature_pool = torch.permute(spatio_temporal_feature_pool, dims=[0, 2, 1])
        # if self.neck_cls_dropout is not None:
        #     spatio_temporal_feature_pool_dropout = self.neck_cls_dropout(spatio_temporal_feature_pool)  # [N, num_segs, 2048]

        # # [N, num_seg, num_class]
        # neck_score = self.neck_cls(spatio_temporal_feature_pool_dropout)

        # [N D T]
        # seg_feature = (backbone_feature + spatio_temporal_feature_pool).contiguous() * masks[:, 0:1, :]
        seg_feature = backbone_feature.contiguous() * masks[:, 0:1, :]
        
        if self.neck_dropout is not None:
            seg_feature = self.neck_dropout(seg_feature)

        return seg_feature, backbone_score, None