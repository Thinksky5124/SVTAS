'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-07 19:25:43
Description: model neck
FilePath: /ETESVS/model/neck.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ETESVSNeck(nn.Module):
    def __init__(self,
                 num_classes=11,
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

        return seg_feature