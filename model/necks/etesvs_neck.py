'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-14 17:06:25
Description: model neck
FilePath: /ETESVS/model/necks/etesvs_neck.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from .memory_layer import RNNConvModule

from ..builder import NECKS

@NECKS.register()
class ETESVSNeck(nn.Module):
    def __init__(self,
                 num_classes=11,
                 num_layers=4,
                 out_channel=64,
                 in_channel=2048,
                 clip_seg_num=30,
                 drop_ratio=0.5,
                 num_memory_layer=2,
                 sample_rate=4):
        super().__init__()
        self.num_layers = num_layers
        self.out_channel = out_channel
        self.clip_seg_num = clip_seg_num
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.sample_rate = sample_rate
        self.num_memory_layer = num_memory_layer
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn_conv = nn.ModuleList([copy.deepcopy(RNNConvModule(in_channels=in_channel,
                 out_channels=in_channel,
                 hidden_channels=in_channel//4,
                 conv_cfg = dict(type='Conv1d'),
                 norm_cfg = dict(type='BN1d', requires_grad=True))) for s in range(num_memory_layer)])
        

        self.dropout = nn.Dropout(p=self.drop_ratio)
        
        self.fc = nn.Linear(self.in_channel, num_classes)

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        for rnn_conv in self.rnn_conv:
            rnn_conv._reset_memory()
        # pass

    def forward(self, x, masks):
        # x.shape = [N * num_segs, 2048, 7, 7]
        x = self.avgpool(x)
        # x.shape = [N * num_segs, 2048, 1, 1]

        # segmentation feature branch
        # [N * num_segs, 2048]
        seg_x = torch.squeeze(x)
        # [N, num_segs, 2048]
        seg_feature = torch.reshape(seg_x, shape=[-1, self.clip_seg_num, seg_x.shape[-1]])

        # [N, 2048, num_segs]
        seg_feature = torch.permute(seg_feature, dims=[0, 2, 1])
        # [N, 2048, num_segs]
        for rnn_conv in self.rnn_conv:
            seg_feature = rnn_conv(seg_feature, masks)

        # recognition branch
        cls_feature = torch.permute(seg_feature, dims=[0, 2, 1])
        cls_feature = torch.reshape(cls_feature, shape=[-1, self.in_channel])
        if self.dropout is not None:
            x = self.dropout(cls_feature)  # [N * num_seg, in_channels]

        score = self.fc(x)  # [N * num_seg, num_class]
        score = torch.reshape(
            score, [-1, self.clip_seg_num, score.shape[1]])  # [N, num_seg, num_class]
        score = torch.mean(score, axis=1)  # [N, num_class]
        cls_score = torch.reshape(score,
                               shape=[-1, self.num_classes])  # [N, num_class]
        return seg_feature, cls_score