'''
Author       : Thyssen Wen
Date         : 2022-06-05 10:35:39
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-06 10:57:05
Description  : Transeger joint network module
FilePath     : /ETESVS/model/heads/joint/transeger_joint_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import HEADS

@HEADS.register()
class TransegerJointNet(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 2, hidden_channels)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_channels, num_classes)
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, img_input, label_feature, masks):
        # img_feature [num_stage N D T]
        # label_feaeture [N U D]
        # masks [N T]
        img_seg_score, img_feature = img_input

        # img_feature [N D T]
        img_feature = img_feature.squeeze(0)

        # joint branch
        # img_feature [N T D]
        img_feature_t = torch.permute(img_feature, [0, 2, 1])
        seq_lens = label_feature.size(1)
        target_lens = img_feature_t.size(1)

        label_feature = label_feature.unsqueeze(2)
        img_feature_t = img_feature_t.unsqueeze(1)

        label_feature = label_feature.repeat(1, 1, target_lens, 1)
        img_feature_t = img_feature_t.repeat(1, seq_lens, 1, 1)

        output = torch.cat((label_feature, img_feature_t), dim=-1)

        output = self.fc1(output)
        output = self.tanh(output)
        joint_score = self.fc2(output)

        joint_score = F.log_softmax(joint_score, dim=-1)

        return img_seg_score, joint_score