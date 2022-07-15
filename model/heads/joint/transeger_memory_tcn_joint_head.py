'''
Author       : Thyssen Wen
Date         : 2022-06-05 10:35:39
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-12 16:39:22
Description  : Transeger temporal convolution network joint network module
FilePath     : /ETESVS/model/heads/joint/transeger_memory_tcn_joint_head.py
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..segmentation.memory_tcn import MemoryDilationResidualLyaer

from ...builder import HEADS

@HEADS.register()
class TransegerMemoryTCNJointNet(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_layers=4,
                 hidden_channels=128,
                 sample_rate=4):
        super().__init__()
        self.sample_rate = sample_rate
        self.layers = nn.ModuleList([copy.deepcopy(MemoryDilationResidualLyaer(2 ** (num_layers-1-i), hidden_channels, hidden_channels)) for i in range(num_layers)])
        self.conv_1x1 = nn.Conv1d(in_channels * 2, hidden_channels, 1)
        self.conv_out = nn.Conv1d(hidden_channels, num_classes, 1)
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.apply(self._clean_buffers)
    
    @staticmethod
    def _clean_buffers(m):
        if issubclass(type(m), MemoryDilationResidualLyaer):
            m._resert_memory()

    def forward(self, img_feature, text_feature, masks):
        # img_feature [N D T]
        # text_feature [N D T]
        # masks [N T]

        masks = masks.unsqueeze(1)[:, :, ::self.sample_rate]

        # joint branch
        text_feature = torch.flip(text_feature, dims=[-1])
        # [N D T]
        output = torch.cat([img_feature, text_feature], dim=1)

        feature = self.conv_1x1(output)
        for layer in self.layers:
            feature = layer(feature, masks)
        # [N C T]
        joint_score = self.conv_out(feature) * masks[:, 0:1, :]

        # [N C T] -> [num_satge N C T]
        outputs = joint_score.unsqueeze(0)
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        return outputs