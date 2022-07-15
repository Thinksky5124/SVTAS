'''
Author       : Thyssen Wen
Date         : 2022-06-05 10:47:08
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-12 15:33:13
Description  : Transeger Loss module
FilePath     : /ETESVS/model/losses/transeger_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import RNNTLoss
from .segmentation_loss import SegmentationLoss

from ..builder import LOSSES

@LOSSES.register()
class TransegerLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 smooth_weight=0.15,
                 seg_loss_weights=1.0,
                 joint_network_loss_weights=1.0,
                 ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.seg_loss_weights = seg_loss_weights
        self.joint_network_loss_weights = joint_network_loss_weights
        self.sample_rate = sample_rate
        self.joint_loss = SegmentationLoss(num_classes=self.num_classes, loss_weight=self.joint_network_loss_weights, 
                sample_rate=self.sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)
        self.img_seg_loss = SegmentationLoss(self.num_classes, ignore_index=self.ignore_index,
                                                loss_weight=seg_loss_weights,
                                                smooth_weight=smooth_weight, sample_rate=sample_rate)
        self.elps = 1e-10

    def forward(self, model_output, input_data):
        # img_seg_score [stage_num, N, C, T]
        # img_extract_score [N, C, T]
        # joint_score [num_satge N C T]
        img_seg_score, joint_score = model_output

        # img backbone label learning
        img_seg_loss = self.img_seg_loss(img_seg_score, input_data)['loss']
        joint_loss = self.joint_loss(joint_score, input_data)["loss"]

        loss = img_seg_loss + joint_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["img_seg_loss"] = img_seg_loss
        loss_dict["joint_loss"] = joint_loss
        return loss_dict