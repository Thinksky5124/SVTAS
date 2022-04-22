'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-22 21:27:18
Description: model neck
FilePath: /ETESVS/model/necks/etesvs_neck.py
'''
import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from .memory_layer import LSTMResidualLayer

from ..builder import NECKS

@NECKS.register()
class ETESVSNeck(nn.Module):
    def __init__(self,
                 num_classes=11,
                 num_layers=1,
                 num_stages=1,
                 in_channel=1280,
                 clip_seg_num=30,
                 drop_ratio=0.5,
                 bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.clip_seg_num = clip_seg_num
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.num_stages = num_stages
        self.bidirectional = bidirectional
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.softmax = nn.Softmax(dim=1)
        self.softmax_neck_score = nn.Softmax(dim=-1)
        self.rnn_conv = LSTMResidualLayer(self.in_channel, self.in_channel, self.num_classes, self.num_layers,
                                dropout=self.drop_ratio, bidirectional=self.bidirectional)

        self.dropout = nn.Dropout(p=self.drop_ratio)
        self.temporal_dropout = nn.Dropout(p=self.drop_ratio)
        
        self.fc = nn.Linear(self.in_channel, num_classes)

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.rnn_conv._reset_memory()
        # pass

    def forward(self, x, masks):
        # x.shape = [N * num_segs, 2048, 7, 7]
        # masks.shape = [N, T]
        t_x = torch.reshape(x, shape=[-1, self.clip_seg_num] + list(x.shape[-3:]))

        # memory branch
        # [N, 2048, num_segs]
        seg_feature, neck_score = self.rnn_conv(t_x, masks)
        neck_score = torch.permute(neck_score, dims=[0, 2, 1])

        # x.shape = [N * num_segs, 2048, 1, 1]
        x = self.avgpool(x)

        # segmentation feature branch
        # [N * num_segs, 2048]
        seg_x = torch.squeeze(x)
        # [N, num_segs, 2048]
        seg_feature = torch.reshape(seg_x, shape=[-1, self.clip_seg_num, seg_x.shape[-1]])

        # [N, 2048, num_segs]
        feature = torch.permute(seg_feature, dims=[0, 2, 1])

        # recognition branch
        cls_feature = torch.permute(feature, dims=[0, 2, 1])
        cls_feature = torch.reshape(cls_feature, shape=[-1, self.in_channel])
        if self.dropout is not None:
            x = self.dropout(cls_feature)  # [N * num_seg, in_channels]

        score = self.fc(x)  # [N * num_seg, num_class]
        backbone_score = torch.reshape(
            score, [-1, self.clip_seg_num, score.shape[1]]).permute([0, 2, 1])  # [N, num_class, num_seg]
        
        return seg_feature, backbone_score, neck_score