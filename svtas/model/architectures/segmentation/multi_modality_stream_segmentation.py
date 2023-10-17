'''
Author       : Thyssen Wen
Date         : 2022-05-03 16:24:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-17 10:24:23
Description  : Multi Modality stream segmentation
FilePath     : /SVTAS/svtas/model/architectures/segmentation/multi_modality_stream_segmentation.py
'''
import torch
import torch.nn as nn

from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import TorchBaseModel

@AbstractBuildFactory.register('model')
class MultiModalityStreamSegmentation(TorchBaseModel):
    def __init__(self,
                 rgb_backbone=None,
                 flow_backbone=None,
                 audio_backbone=None,
                 neck=None,
                 head=None,
                 rgb_backbone_type='3d',
                 flow_backbone_type='3d',
                 audio_backbone_type='pwc',
                 weight_init_cfg=None):
        super().__init__()
        self.rgb_backbone = AbstractBuildFactory.create_factory('model').create(rgb_backbone)
        self.neck = AbstractBuildFactory.create_factory('model').create(neck)
        self.head = AbstractBuildFactory.create_factory('model').create(head)

        if flow_backbone is not None:
            self.flow_backbone = AbstractBuildFactory.create_factory('model').create(flow_backbone)
        else:
            self.flow_backbone = None
        if audio_backbone is not None:
            self.audio_backbone = AbstractBuildFactory.create_factory('model').create(audio_backbone)
        else:
            self.audio_backbone = None

        self.sample_rate = head.sample_rate
        assert rgb_backbone_type in ['3d', '2d']
        assert flow_backbone_type in ['3d', '2d']
        assert audio_backbone_type in ['pwc']
        self.rgb_backbone_type = rgb_backbone_type
        self.flow_backbone_type = flow_backbone_type
        self.audio_backbone_type = audio_backbone_type

    def init_weights(self, init_cfg: dict = {}):
        self.rgb_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
        self.neck.init_weights()
        self.head.init_weights()

        if self.audio_backbone is not None:
            self.audio_backbone.init_weights(child_model=False)
        if self.flow_backbone is not None:
            self.flow_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
    
    def _clear_memory_buffer(self):
        if self.rgb_backbone is not None:
            self.rgb_backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()
        if self.audio_backbone is not None:
            self.audio_backbone._clear_memory_buffer()
        if self.flow_backbone is not None:
            self.flow_backbone._clear_memory_buffer()
    
    def _flow_preprocess(self, x, masks):
        if self.flow_backbone_type == '3d':
            # x.shape=[N,T,C,H,W], for most commonly case
            x = x.transpose(1, 2).contiguous()
            # masks.shape [N, 1, T, 1, 1]
            masks = masks[:, :, ::self.sample_rate].unsqueeze(-1).unsqueeze(-1)
            masks = nn.functional.adaptive_max_pool3d(masks, output_size=[x.shape[2], 1, 1])
        elif self.flow_backbone_type == '2d':
            # x.shape=[N,T,C,H,W], for most commonly case
            masks = masks[:, :, ::self.sample_rate]
            masks = nn.functional.adaptive_max_pool1d(masks, output_size=[x.shape[1]])
            x = torch.reshape(x, [-1] + list(x.shape[2:])).contiguous()
            # x [N * T, C, H, W]
            # masks.shape [N * T, 1, 1, 1]
            masks = torch.reshape(masks.transpose(1, 2), [-1, 1, 1, 1])
        else:
            raise NotImplementedError
        return x, masks

    def _rgb_preprocess(self, x, masks):
        if self.rgb_backbone_type == '3d':
            # x.shape=[N,T,C,H,W], for most commonly case
            x = x.transpose(1, 2).contiguous()
            # masks.shape [N, 1, T, 1, 1]
            masks = masks[:, :, ::self.sample_rate].unsqueeze(-1).unsqueeze(-1)
            masks = nn.functional.adaptive_max_pool3d(masks, output_size=[x.shape[2], 1, 1])
        elif self.rgb_backbone_type == '2d':
            # x.shape=[N,T,C,H,W], for most commonly case
            masks = masks[:, :, ::self.sample_rate]
            masks = nn.functional.adaptive_max_pool1d(masks, output_size=[x.shape[1]])
            x = torch.reshape(x, [-1] + list(x.shape[2:])).contiguous()
            # x [N * T, C, H, W]
            # masks.shape [N * T, 1, 1, 1]
            masks = torch.reshape(masks.transpose(1, 2), [-1, 1, 1, 1])
        else:
            raise NotImplementedError
        return x, masks

    def _audio_preprocess(self, x, masks):
        if self.audio_backbone_type == 'pwc':
            pass
        else:
            raise NotImplementedError
        return x, masks

    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']

        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)
        feature_dict = {}
        if self.audio_backbone is not None:
            audio = input_data['audio']
            audio_x, backbone_masks =  self._flow_preprocess(audio, masks)
            audio_feature = self.audio_backbone(audio_x, backbone_masks)
            feature_dict['audio'] = audio_feature

        if self.flow_backbone is not None:
            flows = input_data['flows']
            flow_x, backbone_masks =  self._flow_preprocess(flows, masks)
            flow_feature = self.flow_backbone(flow_x, backbone_masks)
            feature_dict['flow'] = flow_feature

        if self.rgb_backbone is not None:
            rgb_x, backbone_masks = self._rgb_preprocess(imgs, masks)
            feature = self.rgb_backbone(rgb_x, backbone_masks)
            feature_dict['rgb'] = feature

        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(
                feature_dict, masks[:, :, ::self.sample_rate])
            
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
        # cls_score [N, C, T]
        return {"output":head_score}