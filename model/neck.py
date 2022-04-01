'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-01 21:44:06
Description: model neck
FilePath: /ETESVS/model/neck.py
'''
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .mstcn import SingleStageModel

class ETESVSNeck(nn.Module):
    def __init__(self,
                 num_classes=11,
                 num_stages=1,
                 num_layers=4,
                 out_channel=64,
                 in_channel=2048,
                 clip_seg_num=30,
                 drop_ratio=0.5,
                 sample_rate=4,
                 data_format="NCHW"):
        super().__init__()
        self.num_layers = num_layers
        self.out_channel = out_channel
        self.clip_seg_num = clip_seg_num
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.sample_rate = sample_rate

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"
        self.data_format = data_format
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=self.drop_ratio)
        self.seg_conv = SingleStageModel(num_layers, out_channel, in_channel,
                                       num_classes)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(num_layers, out_channel, num_classes,
                                 num_classes)) for s in range(num_stages - 1)
        ])
        
        self.fc = nn.Linear(self.in_channel, num_classes)

    def init_weights(self):
        pass

    def forward(self, x, masks):
        # x.shape = [N * num_segs, 2048, 7, 7]
        x = self.avgpool(x)
        # x.shape = [N * num_segs, 2048, 1, 1]
        cls_feature = x

        # segmentation feature branch
        # [N * num_segs, 2048]
        seg_x = torch.squeeze(x)
        # [N, num_segs, 2048]
        seg_feature = torch.reshape(seg_x, shape=[-1, self.clip_seg_num, seg_x.shape[-1]])

        # [N, 2048, num_segs]
        seg_feature = torch.permute(seg_feature, dims=[0, 2, 1])

        # segmentation branch
        # seg_feature [N, in_channels, temporal_len]
        # Interploate upsample

        out = self.seg_conv(seg_feature, masks)
        outputs = out.unsqueeze(0)
        # seg_feature [stage_num, N, num_class, temporal_len]
        for s in self.stages:
            out = s(F.softmax(out, dim=1), masks)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        seg_neck_score = outputs

        # recognition branch
        if self.dropout is not None:
            x = self.dropout(cls_feature)  # [N * num_seg, in_channels, 1, 1]

        if self.data_format == 'NCHW':
            x = torch.reshape(x, x.shape[:2])
        else:
            x = torch.reshape(x, x.shape[::3])
        score = self.fc(x)  # [N * num_seg, num_class]
        score = torch.reshape(
            score, [-1, seg_feature.shape[2], score.shape[1]])  # [N, num_seg, num_class]
        score = torch.mean(score, axis=1)  # [N, num_class]
        cls_score = torch.reshape(score,
                               shape=[-1, self.num_classes])  # [N, num_class]
        return seg_feature, seg_neck_score, cls_score