'''
Author       : Thyssen Wen
Date         : 2022-05-04 14:57:21
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-05 16:40:30
Description  : file content
FilePath     : /ETESVS/model/architectures/optical_flow_estimator.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from utils.logger import get_logger

from ..builder import build_backbone
from ..builder import build_neck
from ..builder import build_head

from ..builder import ARCHITECTURE

@ARCHITECTURE.register()
class OpticalFlowEstimation(nn.Module):
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

        if head is not None:
            self.head = build_head(head)
        else:
            self.head = None

        self.init_weights()

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False)
        if self.neck is not None:
            self.neck.init_weights()
        if self.head is not None:
            self.head.init_weights()
    
    def _clear_memory_buffer(self):
        self.backbone._clear_memory_buffer()
        # self.neck._clear_memory_buffer()
        # self.head._clear_memory_buffer()

    def forward(self, input_data):
        flow_imgs = input_data['imgs']

        # feature.shape=[N,T,C,H,W], for most commonly case

        if self.backbone is not None:
            feature = self.backbone(flow_imgs)
        else:
            feature = feature

        # feature [N,T,C,H,W]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(feature)
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N,T,C,H,W]
        # cls_feature [N,T,C,H,W]
        if self.head is not None:
            head_score = self.head(seg_feature)
        else:
            head_score = feature
        # seg_score [N,T,C,H,W]
        return head_score