'''
Author       : Thyssen Wen
Date         : 2022-06-05 10:47:08
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-05 16:08:37
Description  : Transeger Loss module
FilePath     : /ETESVS/model/losses/transeger_loss.py
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import RNNTLoss
from .steam_segmentation_loss import StreamSegmentationLoss

from ..builder import LOSSES

@LOSSES.register()
class TransegerLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 smooth_weight=0.5,
                 img_seg_loss_weights=1.0,
                 img_extract_loss_weights=1.0,
                 joint_network_loss_weights=1.0,
                 ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.img_extract_loss_weights = img_extract_loss_weights
        self.img_seg_loss_weights = img_seg_loss_weights
        self.joint_network_loss_weights = joint_network_loss_weights
        self.sample_rate = sample_rate
        self.elps = 1e-10
        self.rnnt_loss = RNNTLoss(blank=0, reduction="none")
        self.img_seg_loss = StreamSegmentationLoss(self.num_classes, ignore_index=self.ignore_index,
                                                backone_loss_weight=img_extract_loss_weights, head_loss_weight=img_seg_loss_weights,
                                                smooth_weight=smooth_weight, sample_rate=sample_rate)

    
    def forward(self, model_output, input_data):
        # img_extract_score [stage_num, N, C, T]
        # img_seg_score [stage_num, N, C, T]
        # joint_score [N U T+1 C] 
        img_extract_score, img_seg_score, joint_score = model_output
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']

        # img backbone label learning
        img_seg_loss_dict = self.img_seg_loss([img_extract_score, img_seg_score], input_data)

        # joint network label learning
        img_seq_len = torch.tensor(joint_score.shape[1], dtype=torch.int).to(joint_score.device)
        text_seq_len = torch.tensor(joint_score.shape[2], dtype=torch.int).to(joint_score.device)
        joint_loss = self.rnnt_loss(joint_score, labels, text_seq_len, img_seq_len)
        
        img_extract_cls_score_loss = img_seg_loss_dict["backbone_loss"]
        img_seg_score_loss = img_seg_loss_dict["head_loss"]
        joint_loss = self.joint_network_loss_weights * joint_loss
        
        loss = img_extract_cls_score_loss + img_seg_score_loss + joint_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["img_extract_loss"] = img_extract_cls_score_loss
        loss_dict["img_seg_loss"] = img_seg_score_loss
        loss_dict["joint_loss"] = joint_loss
        return loss_dict

