'''
Author       : Thyssen Wen
Date         : 2022-06-05 10:35:39
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-05 15:45:40
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
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_channels, num_classes)

        self.conv1d = nn.Conv1d(in_channels, num_classes)
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, img_feature, label_feature, masks):
        # img_feature [N T D]
        # label_feaeture [N U D]
        # img branch
        img_seg_score = self.conv1d(img_feature) * masks

        # joint branch
        seq_lens = label_feature.size(1)
        target_lens = img_feature.size(1)

        label_feature = label_feature.unsqueeze(2)
        img_feature = img_feature.unsqueeze(1)

        label_feature = label_feature.repeat(1, 1, target_lens, 1)
        img_feature = img_feature.repeat(1, seq_lens, 1, 1)

        output = torch.cat((label_feature, img_feature), dim=-1)

        output = self.fc1(output)
        output = self.tanh(output)
        joint_score = self.fc2(output)

        joint_score = F.log_softmax(joint_score, dim=-1)

        return img_seg_score, joint_score