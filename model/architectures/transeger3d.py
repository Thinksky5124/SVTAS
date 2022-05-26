'''
Author       : Thyssen Wen
Date         : 2022-05-21 11:09:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 19:52:02
Description  :Transgemnter3D framework
FilePath     : /ETESVS/model/architectures/transeger3d.py
'''
import torch
import torch.nn as nn

from ..builder import build_backbone
from ..builder import build_neck
from ..builder import build_head

from ..builder import ARCHITECTURE

@ARCHITECTURE.register()
class Transeger3D(nn.Module):
    def __init__(self,
                 image_backbone=None,
                 text_backbone=None,
                 joint=None,
                 loss=None):
        super().__init__()
        self.image_backbone = build_backbone(image_backbone)
        self.text_backbone = build_backbone(text_backbone)
        self.joint = build_head(joint)

        self.init_weights()
        
        self.sample_rate = joint.sample_rate
    
    def init_weights(self):
        self.image_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        self.text_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        self.joint.init_weights()
    
    def _clear_memory_buffer(self):
        if self.image_backbone is not None:
            self.image_backbone._clear_memory_buffer()
        if self.text_backbone is not None:
            self.text_backbone._clear_memory_buffer()
        if self.joint is not None:
            self.joint._clear_memory_buffer()
    
    def forward(self, input_data):
        if self.training:
            masks = input_data['masks']
            imgs = input_data['imgs']
            labels = input_data['labels']
        else:
            masks = input_data['masks']
            imgs = input_data['imgs']

        ### image encoder
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        # x.shape=[N,T,C,H,W], for most commonly case
        imgs = imgs.transpose(1, 2).contiguous()
        # x [N * T, C, H, W]

        if self.image_backbone is not None:
            # masks.shape [N, 1, T, 1, 1]
            img_backbone_masks = masks[:, :, ::self.sample_rate].unsqueeze(-1).unsqueeze(-1)
            # [N, C, T, H, W]
            img_feature = self.image_backbone(imgs, img_backbone_masks)

            out_channels = img_feature.shape[1]
            # [N, T, C, H, W]
            img_feature = img_feature.transpose(1, 2)
            # [N * T, C, H, W]
            img_feature = torch.reshape(img_feature, shape=[-1, out_channels] + list(img_feature.shape[-2:]))
            # feature [N * T , F_dim, 7, 7]
        else:
            img_feature = imgs
        
        ### text encoder
        if self.training and self.text_backbone is not None:
            # masks.shape [N, 1, T, 1, 1]
            text_backbone_masks = masks[:, :, ::self.sample_rate].unsqueeze(-1).unsqueeze(-1)
            # [N, C, T, H, W]
            text_feature = self.text_backbone(labels, text_backbone_masks)
        else:
            text_feature = None

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.joint is not None:
            head_score = self.joint(img_feature, text_feature, masks)
        else:
            head_score = None
        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]
        return head_score