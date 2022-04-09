'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:10
LastEditors: Thyssen Wen
LastEditTime: 2022-04-08 10:44:11
Description: model framework
FilePath: /ETESVS/model/etesvs.py
'''
import torch
import torch.nn as nn

from .backbone import ETESVSBackBone
from .neck import ETESVSNeck
from .head import ETESVSHead

class ETESVS(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        self.backbone = ETESVSBackBone(**backbone)
        self.neck = ETESVSNeck(**neck)
        self.head = ETESVSHead(**head)
        
        self.init_weights()

        self.sample_rate = head.sample_rate

    def init_weights(self):
        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()
    
    def _clear_memory_buffer(self):
        self.backbone._clear_memory_buffer()
        self.neck._clear_memory_buffer()
        self.head._clear_memory_buffer()

    def forward(self, imgs, masks, idx):
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = None

        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature, cls_score = self.neck(
                feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature
            cls_score = None

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            seg_score = self.head(seg_feature, masks)
        else:
            seg_score = None
        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]
        return seg_score, cls_score