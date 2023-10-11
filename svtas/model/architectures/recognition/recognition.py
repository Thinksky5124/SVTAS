'''
Author       : Thyssen Wen
Date         : 2023-09-25 13:48:29
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 16:19:56
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/recognition/recognition.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..general import SeriousModel
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class Recognition(SeriousModel):
    backbone: nn.Module
    neck: nn.Module
    head: nn.Module
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 weight_init_cfg=dict(
                    backbone=dict(
                    child_model=False,
                    revise_keys=[(r'backbone.', r'')]
                    )),
                 **kwargs) -> None:
        super().__init__(weight_init_cfg=weight_init_cfg, backbone=backbone, neck=neck, head=head, **kwargs)

@AbstractBuildFactory.register('model')
class VideoRocognition(Recognition):
    def __init__(self,
                 architecture_type='2d',
                 backbone=None,
                 neck=None,
                 head=None,
                 weight_init_cfg=dict(
                    backbone=dict(
                    child_model=False,
                    revise_keys=[(r'backbone.', r'')]
                    )),
                 **kwargs) -> None:
        assert architecture_type in ['2d', '3d'], f'Unsupport architecture_type: {architecture_type}!'
        super().__init__(weight_init_cfg=weight_init_cfg, backbone=backbone, neck=neck, head=head, **kwargs)
        self.sample_rate = head.sample_rate
        self.architecture_type = architecture_type
    
    def preprocessing(self, input_data):
        masks = input_data['masks'].unsqueeze(1)
        input_data['masks_m'] = masks

        if self.backbone is not None:
            imgs = input_data['imgs']
            if self.architecture_type == '2d':
                imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:])).contiguous()
                input_data['imgs_m'] = imgs
                input_data['backbone_masks'] = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            elif self.architecture_type == '3d':
                imgs = torch.permute(imgs, dims=[0, 2, 1, 3, 4]).contiguous()
                input_data['imgs_m'] = imgs
                input_data['backbone_masks'] = masks[:, :, ::self.sample_rate].unsqueeze(-1).unsqueeze(-1)
        return input_data
    
    def forward(self, input_data):
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
            seg_feature = self.neck(
                feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = seg_feature
        return {"output":head_score}