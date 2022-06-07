'''
Author       : Thyssen Wen
Date         : 2022-06-05 10:35:39
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-07 10:40:59
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
                 hidden_channels=128,
                 sample_rate=4):
        super().__init__()
        self.sample_rate = sample_rate
        self.fc1 = nn.Linear(in_channels * 2, hidden_channels)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_channels, num_classes)
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, img_feature, text_feature, masks):
        # img_feature [num_stage N D T]
        # text_feature [N D T]
        # masks [N T]

        # img_feature [N D T]
        img_feature = img_feature.squeeze(0)

        # joint branch
        # [N D T] -> [N T D]
        img_feature_t = torch.permute(img_feature, [0, 2, 1])
        text_feature_t = torch.permute(text_feature, [0, 2, 1])
        
        output = torch.cat([img_feature_t, text_feature_t], dim=-1)

        output = self.fc1(output)
        output = self.tanh(output)
        joint_score = self.fc2(output)

        joint_score = F.log_softmax(joint_score, dim=-1)

        # [N T C] -> [N C T]
        joint_score = torch.permute(joint_score, [0, 2, 1]) * masks.unsqueeze(1)[:, 0:1, ::self.sample_rate]

        # [N C T] -> [num_satge N C T]
        outputs = joint_score.unsqueeze(0)
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        return outputs