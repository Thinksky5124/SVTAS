'''
Author       : Thyssen Wen
Date         : 2022-05-21 11:09:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-06 16:50:01
Description  : Transeger framework
FilePath     : /ETESVS/model/architectures/segmentation/transeger.py
'''
import torch
import torch.nn as nn

from ...builder import build_architecture
from ...builder import build_neck
from ...builder import build_head

from ...builder import ARCHITECTURE

@ARCHITECTURE.register()
class Transeger(nn.Module):
    def __init__(self,
                 image_backbone=None,
                 text_backbone=None,
                 joint=None,
                 loss=None):
        super().__init__()
        self.image_backbone = build_architecture(image_backbone)
        self.text_backbone = build_architecture(text_backbone)
        self.joint = build_head(joint)

        self.init_weights()
        
        self.sample_rate = image_backbone.head.sample_rate
        # memory last clip labels
        self.last_clip_labels = None
    
    def init_weights(self):
        self.joint.init_weights()
    
    def _clear_memory_buffer(self):
        if self.image_backbone is not None:
            self.image_backbone._clear_memory_buffer()
        if self.text_backbone is not None:
            self.text_backbone._clear_memory_buffer()
        if self.joint is not None:
            self.joint._clear_memory_buffer()
        self.last_clip_labels = None
    
    def forward(self, input_data):
        if self.training:
            masks = input_data['masks']
            imgs = input_data['imgs']
            labels = input_data['labels']
        else:
            masks = input_data['masks']
            imgs = input_data['imgs']
        
        if self.last_clip_labels is None:
            self.last_clip_labels = labels.detach().clone()
            last_clip_labels = None
        else:
            last_clip_labels = self.last_clip_labels.detach().clone()
            self.last_clip_labels = labels.detach().clone()

        ### image encoder
        # if self.image_backbone is not None:
        #     img_input = {"imgs": imgs, "masks": masks}
        #     img_extract_score, head_feature = self.image_backbone(img_input)
        # else:
        #     img_extract_score = None
        #     head_feature = imgs
        
        ### text encoder
        if self.training and self.text_backbone is not None:
            text_input = {"x": last_clip_labels, "masks": masks}
            text_output = self.text_backbone(text_input)
        else:
            text_output = None

        ### joint img and text
        if self.joint is not None:
            img_seg_score, joint_score = self.joint(head_feature, text_output, masks)
        else:
            img_seg_score = None
            joint_score = None
        # img_seg_score [stage_num, N, C, T]
        # img_extract_score [N, C, T]
        # joint_score [N U T C]
        return img_extract_score, img_seg_score, joint_score