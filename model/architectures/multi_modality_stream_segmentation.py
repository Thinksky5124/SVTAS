'''
Author       : Thyssen Wen
Date         : 2022-05-03 16:24:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-05 19:50:56
Description  : Multi Modality stream segmentation
FilePath     : /ETESVS/model/architectures/multi_modality_stream_segmentation.py
'''
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from utils.logger import get_logger

from ..builder import build_backbone
from ..builder import build_neck
from ..builder import build_head

from ..builder import ARCHITECTURE

@ARCHITECTURE.register()
class MulModStreamSegmentation(nn.Module):
    def __init__(self,
                 rgb_backbone=None,
                 flow_backbone=None,
                 audio_backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        self.rgb_backbone = build_backbone(rgb_backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        if flow_backbone is not None:
            self.flow_backbone = build_backbone(flow_backbone)
        else:
            self.flow_backbone = None
        if audio_backbone is not None:
            self.audio_backbone = build_backbone(audio_backbone)
        else:
            self.audio_backbone = None
        
        self.init_weights()

        self.sample_rate = head.sample_rate

    def init_weights(self):
        self.rgb_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        self.neck.init_weights()
        self.head.init_weights()

        if self.audio_backbone is not None:
            self.audio_backbone.init_weights(child_model=False)
        if self.flow_backbone is not None:
            self.flow_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
    
    def _clear_memory_buffer(self):
        if self.rgb_backbone is not None:
            # self.backbone._clear_memory_buffer()
            pass
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()
        if self.audio_backbone is not None:
            self.audio_backbone._clear_memory_buffer()
        if self.flow_backbone is not None:
            # self.flow_backbone._clear_memory_buffer()
            pass

    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']
        flows = input_data['flows']

        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        if self.audio_backbone is not None:
            audio_feature = self.audio_backbone()
        if self.flow_backbone is not None:
            # x.shape=[N,T,C,H,W], for most commonly case
            flow_x = torch.reshape(flows, [-1] + list(flows.shape[2:]))
            # x [N * T, C, H, W]
            # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            flow_feature = self.flow_backbone(flow_x, backbone_masks)
        else:
            flow_feature = flows

        if self.rgb_backbone is not None:
            # x.shape=[N,T,C,H,W], for most commonly case
            rgb_x = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
            # x [N * T, C, H, W]
            # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.rgb_backbone(rgb_x, backbone_masks)
        else:
            feature = imgs
        
        # fusion
        if self.flow_backbone is not None:
            feature = (flow_feature + feature) / 2.0

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
            head_score = None
        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]
        return backbone_score, head_score