'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 15:52:51
Description: loss function
FilePath     : /ETESVS/model/losses/steam_segmentation_loss.py
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

        self.seg_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.backbone_clip_loss = SoftLabelLoss(self.num_classes, ignore_index=self.ignore_index)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, model_output, masks, labels):
        backbone_score, head_score = model_output
        # seg_score [stage_num, N, C, T]
        # masks [N, T]
        # labels [N, T]

        # classification branch loss
        # backbone smooth label learning

        # backbone label learning
        backbone_cls_score_loss = self.backbone_clip_loss(backbone_score, labels[:, ::self.sample_rate], masks[:, ::self.sample_rate])

        # segmentation branch loss
        seg_loss = 0.
        for p in head_score:
            seg_cls_loss = self.seg_ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            seg_loss += torch.sum(seg_cls_loss / (torch.sum(labels != -100) + self.elps))
            seg_loss += self.smooth_weight * torch.mean(torch.clamp(
                self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)
                ), min=0, max=16) * masks[:, 1:].unsqueeze(1))

        backbone_loss = self.backone_loss_weight * backbone_cls_score_loss
        head_loss = self.head_loss_weight * seg_loss

        # output dict compose
        loss = backbone_loss + head_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["backbone_loss"] = backbone_loss
        loss_dict["head_loss"] = head_loss
        return loss_dict

class SoftLabelLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.elps = 1e-10
    
    def forward(self, score, gt, mask):
        score = torch.mean(score, axis=-1)  # [N, num_class]
        cls_score = torch.reshape(score,
                               shape=[-1, self.num_classes])  # [N, num_class]

        # smooth label learning
        with torch.no_grad():
            device = cls_score.device
            # [N T]
            raw_labels = gt
            # deal label over num_classes
            # [N, 1]
            y = torch.zeros(raw_labels.shape, dtype=raw_labels.dtype, device=device)
            refine_label = torch.where(raw_labels != self.ignore_index, raw_labels, y)
            # [N C T]
            ce_y = F.one_hot(refine_label, num_classes=self.num_classes)

            raw_labels_repeat = torch.tile(raw_labels.unsqueeze(2), dims=[1, 1, self.num_classes])
            ce_y = torch.where(raw_labels_repeat != self.ignore_index, ce_y, torch.zeros(ce_y.shape, device=device, dtype=ce_y.dtype))
            # [N C]
            smooth_label = (torch.sum(ce_y.float(), dim=1) / ce_y.shape[1])

            # [N, 1]
            x = torch.ones((smooth_label.shape[0]), device=device)
            y = torch.zeros((smooth_label.shape[0]), device=device)
            mask = torch.where(torch.sum(smooth_label, dim=1)!=0, x, y)

        cls_loss = torch.mean(self.ce(cls_score, smooth_label) * mask)
        return cls_loss