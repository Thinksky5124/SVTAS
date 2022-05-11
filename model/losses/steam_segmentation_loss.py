'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-11 10:43:19
Description: loss function
FilePath     : /ETESVS/model/losses/steam_segmentation_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss
from .recognition_segmentation_loss import SoftLabelRocgnitionLoss

from ..builder import LOSSES

@LOSSES.register()
class StreamSegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 backone_loss_weight=1.0,
                 head_loss_weight=1.0,
                 smooth_weight=0.15,
                 ignore_index=-100):
        super().__init__()
        self.backone_loss_weight = backone_loss_weight
        self.head_loss_weight = head_loss_weight
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.elps = 1e-10

        self.backbone_clip_loss = SoftLabelRocgnitionLoss(self.num_classes, ignore_index=self.ignore_index)
        self.segmentation_loss = SegmentationLoss(self.num_classes, sample_rate=self.sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)

    def forward(self, model_output, masks, labels, precise_sliding_num):
        backbone_score, head_score = model_output
        # seg_score [stage_num, N, C, T]
        # masks [N, T]
        # labels [N, T]

        # classification branch loss
        # backbone smooth label learning

        # backbone label learning
        backbone_cls_score_loss = self.backbone_clip_loss(backbone_score, labels[:, ::self.sample_rate], masks[:, ::self.sample_rate], precise_sliding_num)['loss']

        # segmentation branch loss
        seg_loss = self.segmentation_loss(head_score, masks, labels, precise_sliding_num)['loss']

        backbone_loss = self.backone_loss_weight * backbone_cls_score_loss
        head_loss = self.head_loss_weight * seg_loss

        # output dict compose
        loss = backbone_loss + head_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["backbone_loss"] = backbone_loss
        loss_dict["head_loss"] = head_loss
        return loss_dict