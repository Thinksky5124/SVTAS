'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:34:16
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 20:16:10
Description  : Segmentation CLIP loss
FilePath     : /ETESVS/model/losses/segmentation_clip_loss.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .steam_segmentation_loss import StreamSegmentationLoss

from ..builder import LOSSES

@LOSSES.register()
class SgementationCLIPLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 smooth_weight=0.15,
                 img_seg_loss_weights=1.0,
                 img_extract_loss_weights=1.0,
                 clip_loss_weight=1.0,
                 ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.img_extract_loss_weights = img_extract_loss_weights
        self.img_seg_loss_weights = img_seg_loss_weights
        self.clip_loss_weight = clip_loss_weight
        self.sample_rate = sample_rate
        self.img_seg_loss = StreamSegmentationLoss(self.num_classes, ignore_index=self.ignore_index,
                                                backone_loss_weight=img_extract_loss_weights, head_loss_weight=img_seg_loss_weights,
                                                smooth_weight=smooth_weight, sample_rate=sample_rate)
        self.clip_loss = CLIPLoss(sample_rate=self.sample_rate, loss_weights=self.clip_loss_weight)
        self.elps = 1e-10
    
    def forward(self, model_output, input_data):
        # img_seg_score [stage_num N C T]
        # img_feature [N C T]
        # text_feature [N C T]
        # img_extract_score [N C T]
        img_feature, text_feature, img_extract_score, img_seg_score = model_output

        # img backbone label learning
        img_seg_loss_dict = self.img_seg_loss([img_extract_score, img_seg_score], input_data)
        clip_loss = self.clip_loss([img_feature, text_feature], input_data)['loss']
        
        img_extract_cls_score_loss = img_seg_loss_dict["backbone_loss"]
        img_seg_score_loss = img_seg_loss_dict["head_loss"]

        loss = img_extract_cls_score_loss + img_seg_score_loss + clip_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["img_extract_loss"] = img_extract_cls_score_loss
        loss_dict["img_seg_loss"] = img_seg_score_loss
        loss_dict["clip_loss"] = clip_loss
        return loss_dict

@LOSSES.register()
class CLIPLoss(nn.Module):
    def __init__(self,
                 sample_rate=4,
                 loss_weights=1.0):
        super().__init__()
        self.loss_weights = loss_weights
        self.sample_rate = sample_rate
        self.ce_i = nn.CrossEntropyLoss(reduction='none')
        self.ce_t = nn.CrossEntropyLoss(reduction='none')
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.elps = 1e-10
    
    def forward(self, model_output, input_data):
        # img_feature [N C T]
        # text_feature [N C T]
        img_feature, text_feature = model_output
        masks, precise_sliding_num = input_data["masks"], input_data['precise_sliding_num']

        b, d_i, T = img_feature.shape
        _, d_t, _ = text_feature.shape

        # [N C T] -> [N T C]
        img_feature = torch.permute(img_feature, dims=[0, 2, 1])

        normal_img_feature = F.normalize(img_feature, dim=-1, p=2.0, eps=self.elps)
        normal_text_feature = F.normalize(text_feature, dim=-2, p=2.0, eps=self.elps)

        # [N T C] * [N C T] -> [N T T]
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * normal_img_feature @ normal_text_feature
        logits_per_text = torch.permute(logits_per_image, dims=[0, 2, 1])

        labels = torch.arange(T).unsqueeze(0).expand(b, -1).to(logits_per_image.device)
        img_loss_no_mean = self.ce_i(logits_per_image.contiguous().view(-1, T), labels.contiguous().view(-1)) * masks[:, ::self.sample_rate].view(-1)
        loss_img = torch.sum(torch.sum(torch.reshape(img_loss_no_mean, shape=[b, T]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(masks != 0.) + self.elps)

        text_loss_no_mean = self.ce_t(logits_per_text.contiguous().view(-1, T), labels.contiguous().view(-1)) * masks[:, ::self.sample_rate].view(-1)
        loss_text = torch.sum(torch.sum(torch.reshape(text_loss_no_mean, shape=[b, T]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(masks != 0.) + self.elps)

        loss = ((loss_img + loss_text) / 2) * self.loss_weights

        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict