'''
Author       : Thyssen Wen
Date         : 2022-10-28 19:55:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 19:19:13
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/segmentation/video/segmentation3d.py
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
class Segmentation3D(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 aligin_head=None,
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
            self.sample_rate = head.sample_rate
        else:
            self.head = None
            self.sample_rate = loss.sample_rate
        
        if aligin_head is not None:
            self.aligin_head = build_head(aligin_head)
        else:
            self.aligin_head = None

        self.init_weights()

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        if self.neck is not None:
            self.neck.init_weights()
        if self.head is not None:
            self.head.init_weights()
        if self.aligin_head is not None:
            self.aligin_head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            self.backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()
        if self.aligin_head is not None:
            self.aligin_head._clear_memory_buffer()

    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']
        
        # masks.shape=[N,T]
        masks = F.adaptive_max_pool1d(masks, imgs.shape[1], return_indices=False)
        masks = masks.unsqueeze(1)

        # x.shape=[N,T,C,H,W], for most commonly case
        imgs = torch.permute(imgs, dims=[0, 2, 1, 3, 4]).contiguous()
        # imgs.shape=[N,C,T,H,W]

        if self.backbone is not None:
             # masks.shape [N, 1, T, 1, 1]
            backbone_masks = masks[:, :, ::self.sample_rate].unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(imgs, backbone_masks)
        else:
            feature = imgs

        # feature [N, F_dim, T , 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature= self.neck(
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
            
        if self.aligin_head is not None:
            head_score = self.aligin_head(head_score, input_data['labels'], masks)
        else:
            head_score = head_score

        return {"output":head_score}