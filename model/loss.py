'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors: Thyssen Wen
LastEditTime: 2022-04-08 11:22:31
Description: loss function
FilePath: /ETESVS/model/loss.py
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ETESVSLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 seg_weight=1.0,
                 cls_weight=1.0,
                 ignore_index=-100):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.elps = 1e-10

        self.seg_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.cls_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.ce_soft = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, seg_score, cls_score, masks, labels):
        # seg_score [stage_num, N, C, T]
        # masks [N, T]
        # labels [N, T]
        # segmentation branch loss
        seg_loss = 0.
        for p in seg_score:
            seg_cls_loss = self.seg_ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            seg_loss += torch.sum(seg_cls_loss / (torch.sum(labels != -100) + self.elps))
            seg_loss += 0.15 * torch.mean(torch.clamp(
                self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)
                ), min=0, max=16) * masks[:, 1:].unsqueeze(1))

        # classification branch loss
        # hard label learning
        # [N T]
        # ce_y = labels[:, ::self.sample_rate]
        # cls_loss = self.cls_ce(cls_score.transpose(2, 1).contiguous().view(-1, self.num_classes), ce_y.view(-1))

        # smooth label learning
        with torch.no_grad():
            device = cls_score.device
            # [N T]
            raw_labels = labels[:, ::self.sample_rate]
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

        cls_loss = torch.mean(self.ce_soft(cls_score, smooth_label) * mask)

        cls_loss = self.cls_weight * cls_loss
        seg_loss = self.seg_weight * seg_loss
        return cls_loss, seg_loss