'''
Author       : Thyssen Wen
Date         : 2022-05-03 16:24:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-04 15:18:31
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
                 backbone=None,
                 opticalflow=None,
                 audio=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        if opticalflow is not None:
            self.opticalflow = build_backbone(opticalflow)
        else:
            self.opticalflow = None
        if audio is not None:
            self.audio = build_backbone(audio)
        else:
            self.audio = None
        
        self.init_weights()

        self.sample_rate = head.sample_rate

    def init_weights(self):
        self.backbone.init_weights(child_model=True)
        self.neck.init_weights()
        self.head.init_weights()

        if isinstance(self.backbone.pretrained, str):
            logger = logger = get_logger("ETESVS")
            load_checkpoint(self, self.backbone.pretrained, strict=False, logger=logger)

        if self.audio is not None:
            self.audio.init_weights(child_model=False)
        if self.opticalflow is not None:
            self.opticalflow.init_weights(child_model=False)
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            # self.backbone._clear_memory_buffer()
            pass
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()
        if self.audio is not None:
            self.audio._clear_memory_buffer()
        if self.opticalflow is not None:
            self.opticalflow._clear_memory_buffer()

    def forward(self, imgs, masks, idx=None):
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        if self.audio is not None:
            audio_feature = self.audio()
        if self.opticalflow is not None:
            flow_imgs = self.opticalflow(imgs)

        # x.shape=[N,T,C,H,W], for most commonly case
        if self.opticalflow is not None:
            # imgs = torch.cat([imgs, flow_imgs], dim=2)
            x = torch.reshape(flow_imgs, [-1] + list(flow_imgs.shape[2:]))
        else:
            x = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        # x [N * T, C, H, W]

        if self.backbone is not None:
             # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(x, backbone_masks)
        else:
            feature = x

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