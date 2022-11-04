'''
Author: Thyssen Wen
Date: 2022-04-27 17:01:33
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 20:48:50
Description: feaeture segmentation model framework
FilePath     : /SVTAS/svtas/model/architectures/segmentation/feature/feature_segmentation3d.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from .....utils.logger import get_logger

from ....builder import build_backbone
from ....builder import build_neck
from ....builder import build_head

from ....builder import ARCHITECTURE

@ARCHITECTURE.register()
class FeatureSegmentation3D(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        if backbone is not None:
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = None
            
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None

        self.head = build_head(head)

        self.init_weights()

        self.sample_rate = head.sample_rate

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=True)
        if self.neck is not None:
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
        feature = input_data['feature']
        
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)
        # feature.shape=[N C T H W]

        if self.backbone is not None:
             # masks.shape [N C T H W]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1)
            feature = self.backbone(feature, backbone_masks)
        else:
            feature = feature

        # feature [N, F_dim, T]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature, backbone_score, neck_score = self.neck(
                feature, masks)
            
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
            head_score = None
        # seg_score [stage_num, N, C, T]
        return {"output":head_score}