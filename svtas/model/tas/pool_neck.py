'''
Author: Thyssen Wen
Date: 2022-05-02 22:15:00
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 18:43:21
Description: avg pooling 3d to 2d neck
FilePath     : /SVTAS/svtas/model/tas/pool_neck.py
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class PoolNeck(nn.Module):
    """Pool Neck Module
    Pooling space information for different input size tensor.
    
    Shape:
    1: [N T C H W] -> [N C T]
    2: [N*T C H W] -> [N C T]
    3: [N P C] -> [N C T]
    """
    def __init__(self,
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
        self.drop_ratio = drop_ratio
        self.need_pool = need_pool
        assert pool_type in ['mean', 'max'], "f{pool_type} doesn't support!"
        
        if pool_type =='mean':
            self.backbone_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'max':
            self.backbone_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError


    def init_weights(self, init_cfg: dict = {}):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, masks):
        # x.shape = [N * num_segs, in_channels, 7, 7] or [N * num_segs, in_channels]
        # masks.shape = [N, C, T]
        feature = x
        if self.dynamic_shape:
            self.clip_seg_num = x.shape[0] // 2

        if len(list(feature.shape)) == 2:
            feature = feature.unsqueeze(-1).unsqueeze(-1)
        elif len(list(feature.shape)) == 5:
            # [N C T H W] -> [N * T, C, H, W]
            feature = feature.transpose(1, 2)
            feature = torch.reshape(feature, [-1] + list(feature.shape[-3:]))
        elif len(list(feature.shape)) == 3:
            # [N C L]
            feature = torch.reshape(feature, [feature.shape[0], feature.shape[1], int(math.sqrt(feature.shape[-1])), int(math.sqrt(feature.shape[-1]))])

        # backbone branch
        # x.shape = [N * num_segs, in_channels, 1, 1]
        backbone_x = self.backbone_pool(feature)

        # [N * num_segs, in_channels]
        backbone_x = torch.squeeze(backbone_x)
        # [N, in_channels, num_segs]
        backbone_feature = torch.reshape(backbone_x, shape=[-1, self.clip_seg_num, backbone_x.shape[-1]]).transpose(1, 2) * masks[:, 0:1, :]

        if self.need_pool is True:
            # [N, in_channels, num_segs]
            seg_feature = backbone_feature
        else:
            # [N, in_channels, num_segs, 7, 7]
            feature = torch.reshape(feature, shape=[-1, self.clip_seg_num] + list(feature.shape[-3:])).transpose(1, 2)
            seg_feature = feature * masks[:, 0:1, :].unsqueeze(-1).unsqueeze(-1)
        
        return seg_feature