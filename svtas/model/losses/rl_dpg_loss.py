'''
Author       : Thyssen Wen
Date         : 2023-03-09 09:58:40
LastEditors  : Thyssen Wen
LastEditTime : 2023-03-15 13:48:04
Description  : file content
FilePath     : /SVTAS/svtas/model/losses/rl_dpg_loss.py
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss
from .dice_loss import dice_loss

from ..builder import LOSSES

@LOSSES.register()
class RLPGSegmentationLoss(SegmentationLoss):
    def __init__(self,
                 num_classes,
                 loss_weight=1,
                 sample_rate=1,
                 smooth=1,
                 exponent=2,
                 ignore_index=-100,
                 class_weight=None):
        super().__init__(num_classes, loss_weight, sample_rate, 0.0, ignore_index, class_weight)
        self.smooth = smooth
        self.exponent = exponent
    
    def _compute_smooth_loss(self, pred, labels, masks, b, precise_sliding_num):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(labels.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (labels != self.ignore_index).long() * masks.long()

        batch_loss = dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index) / precise_sliding_num

        return torch.mean(batch_loss)
    
    def forward(self, model_output, input_data):
        # score shape [stage_num N C T]
        # masks shape [N T]
        head_score = model_output["output"]
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        
        _, b, _, t = head_score.shape

        loss = 0.
        for p in head_score:
            seg_cls_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            with torch.no_grad():
                reward = 1.0 + self._compute_smooth_loss(p, labels, masks, b, precise_sliding_num)
            loss += reward * torch.sum(torch.sum(torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(labels != -100) + self.elps)
        
        loss_dict={}
        loss_dict["loss"] = loss * self.loss_weight
        return loss_dict