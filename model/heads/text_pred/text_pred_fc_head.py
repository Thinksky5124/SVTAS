'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:17:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 20:00:19
Description  : Text Pred Head
FilePath     : /ETESVS/model/heads/text_pred/text_pred_fc_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ...builder import HEADS

@HEADS.register()
class TextPredFCHead(nn.Module):
    def __init__(self,
                 num_classes,
                 clip_seg_num=15,
                 sample_rate=4,
                 drop_ratio=0.5,
                 in_channels=2048,
                 out_feature=False,
                 init_std=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_seg_num = clip_seg_num
        self.drop_ratio = drop_ratio
        self.out_feature = out_feature
        self.init_std = init_std

        self.dropout = nn.Dropout(p=self.drop_ratio)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def init_weights(self):
        normal_init(self.fc, std=self.init_std)

    def _clear_memory_buffer(self):
        pass
    
    def forward(self, feature, masks):
        # feature [N, in_channels, clip_seg_num]
        # mask [N T]

        if self.dropout is not None:
            feature_drop = self.dropout(feature)  # [N, in_channel, num_seg]
        
        feature_drop = torch.reshape(feature_drop.transpose(1, 2), shape=[-1, self.in_channels])

        # [N * num_segs, C]
        score = self.fc(feature_drop)

        # [N, num_class, num_seg]
        score = torch.reshape(
            score, [-1, self.clip_seg_num, self.num_classes]).permute([0, 2, 1])
        score = score * masks.unsqueeze(1)[:, 0:1, ::self.sample_rate]
        # [stage_num, N, C, T]
        score = score.unsqueeze(0)
        score = F.interpolate(
            input=score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        
        if self.out_feature is True:
            return feature, score

        return score