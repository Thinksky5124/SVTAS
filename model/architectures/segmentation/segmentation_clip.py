'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:05:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 20:05:00
Description  : Segmentation model clip way to train
FilePath     : /ETESVS/model/architectures/segmentation/segmentation_clip.py
'''
import torch
import torch.nn as nn

from ...builder import build_architecture
from ...builder import build_neck
from ...builder import build_head

from ...builder import ARCHITECTURE

@ARCHITECTURE.register()
class SegmentationCLIP(nn.Module):
    def __init__(self,
                 image_backbone=None,
                 text_backbone=None,
                 fusion_model=None,
                 loss=None):
        super().__init__()
        self.image_backbone = build_architecture(image_backbone)
        self.text_backbone = build_architecture(text_backbone)
        if fusion_model is not None:
            self.fusion_model = build_head(fusion_model)
        else:
            self.fusion_model = None

        self.init_weights()
        
        self.sample_rate = image_backbone.head.sample_rate
    
    def init_weights(self):
        if self.fusion_model is not None:
            self.fusion_model.init_weights()
    
    def _clear_memory_buffer(self):
        if self.image_backbone is not None:
            self.image_backbone._clear_memory_buffer()
        if self.text_backbone is not None:
            self.text_backbone._clear_memory_buffer()
        if self.fusion_model is not None:
            self.fusion_model._clear_memory_buffer()
    
    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']
        labels = input_data['labels']

        ### text encoder
        if self.text_backbone is not None:
            text_input = {"x": labels, "masks": masks}
            text_output = self.text_backbone(text_input)
        else:
            text_output = labels

        ### image encoder
        if self.image_backbone is not None:
            img_input = {"imgs": imgs, "masks": masks}
            img_output = self.image_backbone(img_input)
        else:
            img_output = imgs
        
        ###
        if self.fusion_model is not None:
            fusion_output = self.fusion_model(img_output, text_output, masks)
        else:
            text_feature = text_output
            img_extract_score, head_output = img_output
            img_feature, img_seg_score = head_output
            fusion_output = img_feature, text_feature, img_extract_score, img_seg_score
        # img_seg_score [stage_num, N, C, T]
        # img_feature [N, C, T]
        # text_feature [N, C, T]
        # img_extract_score [N, C, T]
        return fusion_output