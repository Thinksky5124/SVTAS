'''
Author       : Thyssen Wen
Date         : 2022-11-07 14:52:20
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-07 15:17:10
Description  : file content
FilePath     : /SVTAS/svtas/model/necks/ipb_fusion_neck.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS

@NECKS.register()
class IPBFusionNeck(nn.Module):
    def __init__(self,
                 gop_size=15,
                 spatial_expan_mode='bilinear'):
        super().__init__()
        self.gop_size = gop_size
        self.spatial_expan_mode = spatial_expan_mode

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, masks):
        rgb_feature = x[:, :1024, :]
        of_feature = x[:, 1024:, :]
        # x [N C T]
        i_feature = rgb_feature[:, :, ::self.gop_size]
        expand_i_feature = F.interpolate(i_feature.unsqueeze(0), size=[of_feature.shape[-2], of_feature.shape[-1]], mode=self.spatial_expan_mode).squeeze(0)
        feature = torch.concat([expand_i_feature, of_feature], dim=1)
        return feature * masks