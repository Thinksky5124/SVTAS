'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-22 22:14:46
Description: loss function
FilePath     : /SVTAS/svtas/model/losses/steam_segmentation_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss
from .recognition_segmentation_loss import RecognitionSegmentationLoss

from ..builder import LOSSES

@LOSSES.register()
class StreamSegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone_sample_rate=4,
                 head_sample_rate=4,
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
        self.backbone_sample_rate = backbone_sample_rate
        self.head_sample_rate = head_sample_rate
        self.elps = 1e-10

        self.backbone_clip_loss = SegmentationLoss(num_classes=self.num_classes, loss_weight=self.backone_loss_weight, 
                sample_rate=self.backbone_sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)
        self.segmentation_loss = SegmentationLoss(num_classes=self.num_classes, loss_weight=self.head_loss_weight, 
                sample_rate=self.head_sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)

    def forward(self, model_output, input_data):
        backbone_score, head_score = model_output["backbone_score"], model_output["output"]
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        # seg_score [stage_num, N, C, T]
        # masks [N, T]
        # labels [N, T]

        # classification branch loss
        # backbone smooth label learning

        # backbone label learning
        backbone_loss_info = {"masks": masks[:, ::self.backbone_sample_rate], "labels": labels[:, ::self.backbone_sample_rate], "precise_sliding_num": precise_sliding_num}
        backbone_loss = self.backbone_clip_loss({"output":backbone_score.unsqueeze(0)}, backbone_loss_info)['loss']

        # segmentation branch loss
        head_loss_info = {"masks": masks, "labels": labels, "precise_sliding_num": precise_sliding_num}
        head_loss = self.segmentation_loss({"output":head_score}, head_loss_info)['loss']

        # output dict compose
        loss = backbone_loss + head_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["backbone_loss"] = backbone_loss
        loss_dict["head_loss"] = head_loss
        return loss_dict