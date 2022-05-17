'''
Author       : Thyssen Wen
Date         : 2022-05-11 20:32:13
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-17 19:35:06
Description  : MoViNet Head
FilePath     : /ETESVS/model/heads/movinet_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from ..backbones.movinet import ConvBlock3D, Swish, TemporalCGAvgPool3D, CausalModule

from ..builder import HEADS

@HEADS.register()
class MoViNetHead(nn.Module):
    def __init__(self,
                 num_classes,
                 causal=True,
                 tf_like=True,
                 conv_type="3d",
                 clip_seg_num=15,
                 sample_rate=4,
                 drop_ratio=0.5,
                 in_channels=2048,
                 hidden_channels=2048):
        super().__init__()
        self.causal = causal
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_seg_num = clip_seg_num
        self.drop_ratio = drop_ratio
        if sample_rate % 2 != 0:
            raise NotImplementedError
        
        # if causal:
        #     self.cgap = TemporalCGAvgPool3D()

        self.avgpool = nn.AdaptiveAvgPool3d((self.clip_seg_num, 1, 1))

        # dense9
        self.dense = ConvBlock3D(self.in_channels,
                    self.hidden_channels,
                    kernel_size=(1, 1, 1),
                    tf_like=tf_like,
                    causal=causal,
                    conv_type=conv_type,
                    bias=True)
        self.swish = Swish()
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        # dense10d
        self.cls = ConvBlock3D(self.hidden_channels,
                    self.num_classes,
                    kernel_size=(1, 1, 1),
                    tf_like=tf_like,
                    causal=causal,
                    conv_type=conv_type,
                    bias=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)

    def _clear_memory_buffer(self):
        self.apply(self._clean_activation_buffers)

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()
    
    def forward(self, feature, masks):
        # segmentation branch
        # feature [N * num_segs, 1280, 7, 7]

        # feature [N * num_segs, 1280, 1, 1]
        feature = self.avgpool(feature)

        # [N * num_segs, C]
        feature = self.swish(self.dense(feature))
        feature = self.dropout(feature)
        score = self.cls(feature)

        # [N, num_class, num_seg]
        score = torch.reshape(
            score, [-1, self.clip_seg_num, self.num_classes]).permute([0, 2, 1])
        score = score * masks[: ,0:1, :]

        score = score.unsqueeze(0)
        score = F.interpolate(
            input=score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        return score