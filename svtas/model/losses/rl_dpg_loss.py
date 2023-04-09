'''
Author       : Thyssen Wen
Date         : 2023-03-09 09:58:40
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-09 20:40:45
Description  : file content
FilePath     : /SVTAS/svtas/model/losses/rl_dpg_loss.py
'''
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss
from ..builder import LOSSES

def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return num / den

@LOSSES.register()
class RLPGSegmentationLoss(SegmentationLoss):
    def __init__(self,
                 num_classes,
                 loss_weight=1,
                 smooth_weight=0.15,
                 sample_rate=1,
                 smooth=1e-5,
                 exponent=2,
                 beta_1=4,
                 beta_2=-1,
                 ignore_index=-100,
                 class_weight=None):
        super().__init__(num_classes, loss_weight, sample_rate, smooth_weight, ignore_index, class_weight)
        self.smooth = smooth
        self.exponent = exponent
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def _compute_dice_loss(self, pred, labels, masks, b, precise_sliding_num):
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
            ignore_index=self.ignore_index)

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
            seg_cls_loss = torch.sum(torch.sum(torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(labels != -100) + self.elps)
            if(self.smooth_weight> 0.0):
                seg_cls_loss += self.smooth_weight * self._compute_smooth_loss(p, labels, masks, b, precise_sliding_num)
            loss += seg_cls_loss
        with torch.no_grad():
            reward = self.beta_1 ** (self._compute_dice_loss(head_score[-1], labels, masks, b, precise_sliding_num)) - self.beta_2
        loss_dict={}
        loss_dict["loss"] = self.loss_weight * loss * reward
        return loss_dict