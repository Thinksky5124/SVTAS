'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:10
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-04 15:17:40
Description: etesvs model framework
FilePath     : /ETESVS/model/architectures/segmentation/stream_segmentation2d.py
'''
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from utils.logger import get_logger

from ...builder import build_backbone
from ...builder import build_neck
from ...builder import build_head

from ...builder import ARCHITECTURE

@ARCHITECTURE.register()
class StreamSegmentation2D(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        self.init_weights()
        
        self.sample_rate = head.sample_rate

    def init_weights(self):
        self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        self.neck.init_weights()
        self.head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            self.backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()

    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']

        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        # x.shape=[N,T,C,H,W], for most commonly case
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        # x [N * T, C, H, W]

        if self.backbone is not None:
             # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(imgs, backbone_masks)
        else:
            feature = imgs

        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature, backbone_score = self.neck(
                feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature
            backbone_score = None

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = seg_feature
        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]
        return backbone_score, head_score