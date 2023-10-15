'''
Author       : Thyssen Wen
Date         : 2022-12-22 21:24:44
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-15 10:15:46
Description  : file content
FilePath     : /SVTAS/svtas/model/tas/task_fuion_neck.py
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class TaskFusionPoolNeck(nn.Module):
    """Task Fusion Neck Module
    Fuion classification and pooling space information for different input size tensor.
    
    Shape:
    1: [N T C H W] -> [N T C]
    2: [N*T C H W] -> [N T C]
    3: [N P C] -> [N T C]
    """
    def __init__(self,
                 num_classes=11,
                 in_channels=1280,
                 clip_seg_num=None,
                 drop_ratio=0.5,
                 need_pool=True,
                 pool_type='mean'):
        super().__init__()
        self.dynamic_shape = False
        if clip_seg_num is None:
            self.dynamic_shape = True
        self.clip_seg_num = clip_seg_num
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.need_pool = need_pool
        assert pool_type in ['mean', 'max'], "f{pool_type} doesn't support!"
        
        if pool_type =='mean':
            self.backbone_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'max':
            self.backbone_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError

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
        if self.dynamic_shape:
            self.clip_seg_num = x.shape[-1]

        if len(list(feature.shape)) == 2:
            feature = feature.unsqueeze(-1).unsqueeze(-1)
        elif len(list(feature.shape)) == 5:
            # [N C T H W] -> [N * T, C, H, W]
            feature = feature.transpose(1, 2)
            feature = torch.reshape(feature, [-1] + list(feature.shape[-3:]))
        elif len(list(feature.shape)) == 3 and feature.shape[-1] != self.clip_seg_num:
            # [N C L]
            feature = torch.reshape(feature, [feature.shape[0], feature.shape[1], int(math.sqrt(feature.shape[-1])), int(math.sqrt(feature.shape[-1]))])
        
        if self.need_pool is True:
            copy_feature = feature.clone()
            # backbone branch
            # x.shape = [N * num_segs, in_channels, 1, 1]
            backbone_x = self.backbone_pool(copy_feature)

            # [N * num_segs, in_channels]
            backbone_x = torch.squeeze(backbone_x)
            # [N, in_channels, num_segs]
            backbone_feature = torch.reshape(backbone_x, shape=[-1, self.clip_seg_num, backbone_x.shape[-1]]).transpose(1, 2) * masks[:, 0:1, :]
            # get pass feature
            backbone_cls_feature = torch.reshape(backbone_feature.transpose(1, 2), shape=[-1, self.in_channels])

        else:
            # backbone branch
            # x.shape = [N * num_segs, in_channels, 1, 1]
            backbone_x = self.backbone_pool(feature)

            # [N * num_segs, in_channels]
            backbone_x = torch.squeeze(backbone_x)
            # [N, in_channels, num_segs]
            backbone_feature = torch.reshape(backbone_x, shape=[-1, self.clip_seg_num, backbone_x.shape[-1]]).transpose(1, 2) * masks[:, 0:1, :]
            copy_backbone_feature = feature.clone()
            
            # get pass feature
            backbone_cls_feature = torch.reshape(copy_backbone_feature.transpose(1, 2), shape=[-1, self.in_channels])

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