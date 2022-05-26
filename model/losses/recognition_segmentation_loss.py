'''
Author: Thyssen Wen
Date: 2022-04-29 10:56:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 19:49:42
Description: Action recognition model loss
FilePath     : /ETESVS/model/losses/recognition_segmentation_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss

from ..builder import LOSSES

@LOSSES.register()
class RecognitionSegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 label_mode='soft',
                 sample_rate=4,
                 loss_weight=1.0,
                 ignore_index=-100):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.label_mode = label_mode
        self.elps = 1e-10

        if self.label_mode in ["soft"]:
            self.criteria = SoftLabelRocgnitionLoss(self.num_classes, ignore_index=self.ignore_index)
        elif self.label_mode in ['hard']:
            self.criteria = SegmentationLoss(self.num_classes, sample_rate=self.sample_rate, ignore_index=self.ignore_index)
        else:
            raise NotImplementedError

    def forward(self, model_output, input_data):
        score = model_output
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        # seg_score [stage_num, N, C, T]
        # masks [N, T]
        # labels [N, T]

        # classification branch loss
        if self.label_mode in ["soft"]:
            labels = labels[:, ::self.sample_rate]
            masks = masks[:, ::self.sample_rate]
        elif self.label_mode in ['hard']:
            labels = labels[:, ::self.sample_rate]
            labels = torch.repeat_interleave(labels, self.sample_rate, dim=-1)
            masks = masks[:, ::self.sample_rate]
            masks = torch.repeat_interleave(masks, self.sample_rate, dim=-1)

        loss_info = {"masks": masks, "labels": labels, "precise_sliding_num": precise_sliding_num}
        loss = self.criteria(score, loss_info)['loss']

        loss = self.loss_weight * loss

        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict

@LOSSES.register()
class SoftLabelRocgnitionLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.elps = 1e-10
    
    def forward(self, score, input_data):
        gt_mask, gt, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        # gt_mask [N, T]
        score = torch.sum(score * gt_mask.unsqueeze(1), axis=-1) / (torch.sum(gt_mask.unsqueeze(1), dim=-1) + self.elps)  # [N, num_class]
        cls_score = torch.reshape(score, shape=[-1, self.num_classes])  # [N, num_class]

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

        loss = torch.sum(
            (self.ce(cls_score, smooth_label) * mask) / (precise_sliding_num + self.elps)
            ) / (torch.sum(mask) + self.elps)

        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict