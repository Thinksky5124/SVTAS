'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:05:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-11 15:10:11
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
                 loss=None):
        super().__init__()
        self.image_backbone = build_architecture(image_backbone)
        self.text_backbone = build_architecture(text_backbone)

        self.init_weights()
        
        self.sample_rate = image_backbone.head.sample_rate
    
    def init_weights(self):
        pass
    
    def _clear_memory_buffer(self):
        if self.image_backbone is not None:
            self.image_backbone._clear_memory_buffer()
        if self.text_backbone is not None:
            self.text_backbone._clear_memory_buffer()
    
    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']
        labels = input_data['labels']

        ### text encoder
        if self.text_backbone is not None:
            text_input = {"x": labels, "masks": masks}
            text_output = self.text_backbone(text_input)
            text_feature = text_output
        else:
            text_feature = labels

        ## image encoder
        if self.image_backbone is not None:
            img_input = {"imgs": imgs, "masks": masks}
            img_output = self.image_backbone(img_input)
            img_extract_score, head_output = img_output
            img_seg_score, img_feature = head_output
        else:
            img_extract_score = None
            img_seg_score = None
            img_feature = imgs
        
        # img_seg_score [stage_num, N, C, T]
        # img_feature [N, C, T]
        # text_feature [N, C, T]
        # img_extract_score [N, C, T]
        return img_feature, text_feature, img_extract_score, img_seg_score