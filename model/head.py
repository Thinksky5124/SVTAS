'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:13
LastEditors: Thyssen Wen
LastEditTime: 2022-04-01 21:45:33
Description: model head
FilePath: /ETESVS/model/head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ETESVSHead(nn.Module):
    def __init__(self,
                 num_classes=48,
                 num_stages=1,
                 num_layers=4,
                 num_f_maps=64,
                 cls_in_channels=2048,
                 seg_in_channels=2048,
                 sample_rate=4,
                 drop_ratio=0.5,
                 data_format="NCHW"):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps
        self.cls_in_channels = cls_in_channels
        self.seg_in_channels = seg_in_channels
        self.sample_rate = sample_rate
        self.drop_ratio = drop_ratio

    def init_weights(self):
        pass
    
    def forward(self, seg_feature, seg_neck_score, masks):
        # segmentation branch
        # seg_feature [num_stage, N, in_channels, temporal_len]
        seg_score = seg_neck_score
        return seg_score