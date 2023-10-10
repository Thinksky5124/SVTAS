'''
Author       : Thyssen Wen
Date         : 2022-11-22 13:59:48
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 09:39:50
Description  : ref:https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/losses/dice_loss.py
FilePath     : /SVTAS/svtas/model/losses/dice_loss.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss

from svtas.utils import AbstractBuildFactory

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

    return 1 - num / den


@AbstractBuildFactory.register('loss')
class DiceSegmentationLoss(SegmentationLoss):
    """DiceLoss.
    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
    """
    def __init__(self,
                 num_classes,
                 loss_weight=1,
                 sample_rate=1,
                 smooth=1,
                 exponent=2,
                 smooth_weight=0.5,
                 ignore_index=-100,
                 class_weight=None):
        super().__init__(num_classes, 
                         loss_weight,
                         sample_rate,
                         smooth_weight,
                         ignore_index,
                         class_weight)
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