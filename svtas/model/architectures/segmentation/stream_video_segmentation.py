'''
Author       : Thyssen Wen
Date         : 2023-09-25 14:34:10
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 19:24:33
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/segmentation/stream_video_segmentation.py
'''
import torch
import torch.nn as nn

from svtas.utils import AbstractBuildFactory
from ..recognition import VideoRocognition

@AbstractBuildFactory.register('model')
class StreamVideoSegmentation(VideoRocognition):
    def __init__(self,
                 architecture_type='2d',
                 addition_loss_pos=None,
                 backbone=None,
                 neck=None,
                 head=None,
                 weight_init_cfg=dict(
                    backbone=dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]
                    ))):
        super().__init__(architecture_type=architecture_type, weight_init_cfg=weight_init_cfg, backbone=backbone, neck=neck, head=head)
        self.addition_loss_pos = addition_loss_pos
    
    def forward_with_backbone_loss(self, input_data):
        input_data = self.preprocessing(input_data)

        masks = input_data['masks_m']
        imgs = input_data['imgs_m']
        
        if self.backbone is not None:
            backbone_masks = input_data['backbone_masks']
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
        return {"output":head_score, "backbone_score":backbone_score}
    
    def forward_with_backbone_neck_loss(self, input_data):
        input_data = self.preprocessing(input_data)

        masks = input_data['masks_m']
        imgs = input_data['imgs_m']
        
        if self.backbone is not None:
            backbone_masks = input_data['backbone_masks']
            feature = self.backbone(imgs, backbone_masks)
        else:
            feature = imgs

        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature, backbone_score, neck_score = self.neck(
                feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature
            backbone_score = None
            neck_score = None

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = seg_feature
        return {"output":head_score, "backbone_score":backbone_score, "neck_score":neck_score}
    
    def forward(self, input_data):
        if self.addition_loss_pos is None:
            return super().forward(input_data)
        elif self.addition_loss_pos == 'with_backbone_loss':
            return self.forward_with_backbone_loss(input_data)
        elif self.addition_loss_pos == 'with_backbone_neck_loss':
            return self.forward_with_backbone_neck_loss(input_data)
        else:
            raise NotImplementedError