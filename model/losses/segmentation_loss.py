'''
Author: Thyssen Wen
Date: 2022-04-27 20:01:21
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-11 10:31:51
Description: MS-TCN loss model
FilePath     : /ETESVS/model/losses/segmentation_loss.py
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register()
class SegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=1,
                 smooth_weight=0.15,
                 ignore_index=-100):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.elps = 1e-10
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, head_score, masks, labels, precise_sliding_num):
        precise_sliding_num_view = torch.repeat_interleave(precise_sliding_num.to(head_score.device), head_score.shape[-1], dim=-1).view(-1)
        precise_sliding_num_smooth = torch.repeat_interleave(precise_sliding_num.to(head_score.device).unsqueeze(-1), head_score.shape[-1] - 1, dim=-1).unsqueeze(1)
        loss = 0.
        for p in head_score:
            seg_cls_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1)) / (precise_sliding_num_view + self.elps)
            loss += torch.sum(seg_cls_loss / (torch.sum(labels != -100) + self.elps))
            loss += self.smooth_weight * torch.mean(torch.clamp(
                self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)
                ), min=0, max=16) * masks[:, 1:].unsqueeze(1) / (precise_sliding_num_smooth + self.elps))
        
        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict