'''
Author       : Thyssen Wen
Date         : 2022-11-07 14:52:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 15:22:41
Description  : file content
FilePath     : /SVTAS/svtas/model/necks/ipb_fusion_neck.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class IPBFusionNeck(nn.Module):
    def __init__(self,
                 gop_size=15,
                 spatial_expan_mode='bilinear',
                 parse_method='combine',
                 fusion_neck_module=None):
        super().__init__()
        assert parse_method in ['combine', 'separate']
        self.parse_method = parse_method
        self.gop_size = gop_size
        self.spatial_expan_mode = spatial_expan_mode
        if fusion_neck_module is not None:
            self.fusion_neck_module = AbstractBuildFactory.create_factory('model').create(fusion_neck_module)
        else:
            self.fusion_neck_module = None

    def init_weights(self, init_cfg: dict = {}):
        pass

    def _clear_memory_buffer(self):
        pass
    
    def _parse_feature(self, x):
        if self.parse_method == 'combine':
            rgb_feature = x[:, :1024, :]
            of_feature = x[:, 1024:, :]
            rgb_feature = rgb_feature[:, :, ::self.gop_size]
            rgb_feature = F.interpolate(rgb_feature.unsqueeze(0), size=[of_feature.shape[-2], of_feature.shape[-1]], mode=self.spatial_expan_mode).squeeze(0)
        elif self.parse_method == 'separate':
            rgb_feature = x['rgb']
            of_feature = x['flow']
            rgb_feature = F.interpolate(rgb_feature, size=[of_feature.shape[-3], of_feature.shape[-2], of_feature.shape[-1]], mode=self.spatial_expan_mode)
        else:
            raise NotImplementedError
        return rgb_feature, of_feature

    def forward(self, x, masks):
        rgb_feature, of_feature = self._parse_feature(x)
        feature = torch.concat([rgb_feature, of_feature], dim=1)
        if self.fusion_neck_module is not None:
            return self.fusion_neck_module(feature, masks)
        else:
            return feature * masks