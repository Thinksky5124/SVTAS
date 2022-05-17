'''
Author       : Thyssen Wen
Date         : 2022-05-17 19:20:01
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-17 20:23:37
Description  : Feature Extract Head
FilePath     : /ETESVS/model/heads/feature_extract_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS

@HEADS.register()
class FeatureExtractHead(nn.Module):
    def __init__(self,
                 in_channels=2048,
                 clip_seg_num=32,
                 sample_rate=1,
                 pool_space=True,
                 in_format="N,C,T,H,W",
                 out_format="NCT"):
        super().__init__()
        assert out_format in ["NCT", "NTC"], "Unsupport output format!"
        assert in_format in ["N,C,T,H,W", "N*T,C,H,W", "N*T,C"], "Unsupport input format!"
        self.clip_seg_num = clip_seg_num
        self.in_channels = in_channels
        self.out_format = out_format
        self.sample_rate = sample_rate
        self.pool_space = pool_space
        self.in_format = in_format
        
        if self.pool_space:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, masks):
        feature = x

        if self.in_format in ["N,C,T,H,W"]:
            # [N, in_channels, clip_seg_num, 7, 7] -> [N * clip_seg_num, in_channels, 7, 7]
            feature = torch.reshape(feature.transpose(1, 2), [-1, self.in_channels] + list(feature.shape[-2:]))
        elif self.in_format in ["N*T,C"]:
            # [N * clip_seg_num, in_channels] -> [N * clip_seg_num, in_channels, 1, 1]
            feature = feature.unsqueeze(-1).unsqueeze(-1)

        # feature.shape = [N * clip_seg_num, in_channels, 1, 1]
        if self.avg_pool is not None:
            feature = self.avg_pool(feature)
        # [N * num_segs, in_channels]
        feature = feature.squeeze(-1).squeeze(-1)
        # [N, in_channels, num_segs]
        feature = torch.reshape(feature, shape=[-1, self.clip_seg_num, feature.shape[-1]]).transpose(1, 2) * masks[:, 0:1, ::self.sample_rate]

        # [stage_num, N, C, T]
        feature = feature.unsqueeze(0)
        feature = F.interpolate(
            input=feature,
            scale_factor=[1, self.sample_rate],
            mode="nearest").squeeze(0)
            
        if self.out_format in ["NTC"]:
            feature = torch.permute(feature, dims=[0, 2, 1])

        return feature