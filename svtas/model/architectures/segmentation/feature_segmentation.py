'''
Author: Thyssen Wen
Date: 2022-04-27 17:01:33
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 14:56:25
Description: feaeture segmentation model framework
FilePath     : /SVTAS/svtas/model/architectures/segmentation/feature_segmentation.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils.logger import get_logger
from svtas.utils import AbstractBuildFactory
from ..general import SeriousModel

@AbstractBuildFactory.register('architecture')
class FeatureSegmentation(SeriousModel):
    backbone: nn.Module
    neck: nn.Module
    head: nn.Module
    def __init__(self,
                 architecture_type='1d',
                 backbone=None,
                 neck=None,
                 head=None,
                 weight_init_cfg=dict(
                    backbone=dict(
                    child_model=True))):
        assert architecture_type in ['1d', '3d'], f'Unsupport architecture_type: {architecture_type}!'
        super().__init__(weight_init_cfg=weight_init_cfg, backbone=backbone, neck=neck, head=head)
        self.sample_rate = head.sample_rate
        self.architecture_type = architecture_type

    def preprocessing(self, input_data):
        masks = input_data['masks'].unsqueeze(1)
        input_data['masks'] = masks
        
        if self.backbone is not None:
            input_data['backbone_masks'] = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1)
            if self.architecture_type == '1d':
                feature = torch.permute(feature, dims=[0, 2, 1]).contiguous()
                input_data['feature'] = feature
            elif self.architecture_type == '3d':
                pass
        return input_data
    
    def forward(self, input_data):
        input_data = self.preprocessing(input_data)
        masks = input_data['masks']
        feature = input_data['feature']

        if self.backbone is not None:
             # masks.shape [N C T]
            backbone_masks = input_data['backbone_masks']
            feature = self.backbone(feature, backbone_masks)
        else:
            feature = feature

        # feature [N, F_dim, T]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(
                feature, masks)
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = None
        # seg_score [stage_num, N, C, T]
        return {"output":head_score}