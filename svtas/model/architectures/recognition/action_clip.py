'''
Author       : Thyssen Wen
Date         : 2022-10-26 09:57:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 11:28:48
Description  : CLIP achitectures
FilePath     : /SVTAS/svtas/model/architectures/recognition/action_clip.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner import load_state_dict
from collections import OrderedDict
import re

from svtas.utils.logger import get_logger

from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import TorchModel

@AbstractBuildFactory.register('architecture')
class ActionCLIP(TorchModel):
    def __init__(self,
                 pretrained=None,
                 image_prompt=None,
                 text_prompt=None,
                 fusion_neck=None,
                 head=None,
                 is_feature_extract=False,
                 weight_init_cfg=dict(
                     sample_rate=1)):
        super().__init__()
        self.pretrained = pretrained
        self.is_feature_extract = is_feature_extract
        
        if image_prompt is not None:
            self.image_prompt = AbstractBuildFactory.create_factory('model').create(image_prompt)
        else:
            self.image_prompt = None
            
        if text_prompt is not None:
            text_prompt['clip_model'] = self.image_prompt
            self.text_prompt = AbstractBuildFactory.create_factory('model').create(text_prompt)
        else:
            self.text_prompt = None

        if fusion_neck is not None:
            self.fusion_neck = AbstractBuildFactory.create_factory('model').create(fusion_neck)
        else:
            self.fusion_neck = None
        
        if head is not None:
            self.head = AbstractBuildFactory.create_factory('model').create(head)
            self.sample_rate = head.sample_rate
        else:
            self.head = None
            self.sample_rate = weight_init_cfg['sample_rate']
    
        self.init_weights()

    def init_weights(self):
        if isinstance(self.pretrained, str):
            def revise_keys_fn(state_dict, revise_keys=[(r'module.', r'')]):
                # strip prefix of state_dict
                metadata = getattr(state_dict, '_metadata', OrderedDict())
                for p, r in revise_keys:
                    state_dict = OrderedDict(
                        {re.sub(p, r, k): v
                        for k, v in state_dict.items()})
                # Keep metadata in state_dict
                state_dict._metadata = metadata
                return state_dict

            logger  = get_logger("SVTAS")
            checkpoint = torch.load(self.pretrained)
            load_state_dict(self.image_prompt, checkpoint['model_state_dict'], strict=False, logger=logger)
            if self.fusion_neck is not None:
                revise_state_dict = revise_keys_fn(checkpoint['fusion_model_state_dict'])
                load_state_dict(self.fusion_neck, revise_state_dict, strict=False, logger=logger)
        else:
            if self.image_prompt is not None:
                self.image_prompt.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])
            if self.text_prompt is not None:
                self.text_prompt.init_weights()
            if self.fusion_neck is not None:
                self.fusion_neck.init_weights()
            if self.head is not None:
                self.head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.image_prompt is not None:
            self.image_prompt._clear_memory_buffer()
        if self.text_prompt is not None:
            self.text_prompt._clear_memory_buffer()
        if self.fusion_neck is not None:
            self.fusion_neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()

    def forward(self, input_data):

        masks = input_data['masks']
        imgs = input_data['imgs']
        labels = input_data['labels']

        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        # x.shape=[N,T,C,H,W], for most commonly case
        b, t, _, _, _ = imgs.shape
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        # x [N * T, C, H, W]

        if self.text_prompt is not None and self.is_feature_extract is False:
            text_embedding = self.text_prompt(labels, masks)
        else:
            text_embedding = labels

        if self.image_prompt is not None:
             # masks.shape [N * T, 1, 1, 1]
            imgs_masks = masks[:, :, ::self.sample_rate]
            image_embedding = self.image_prompt.encode_image(imgs)
            if self.image_prompt.need_spatial is True:
                image_embedding = image_embedding.permute([0, 2, 1]).reshape(-1, image_embedding.shape[-1], int(math.sqrt(image_embedding.shape[1])), int(math.sqrt(image_embedding.shape[1])))
            else:
                image_embedding = image_embedding.view(b, t, -1).permute([0, 2, 1]) * imgs_masks
        else:
            image_embedding = imgs

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.fusion_neck is not None and self.is_feature_extract is False:
            text_feature, neck_feature = self.fusion_neck(image_embedding, text_embedding, masks)
        else:
            neck_feature = image_embedding
            text_feature = text_embedding
        
        if self.head is not None:
            head_score = self.head(image_embedding, masks)
        else:
            head_score = image_embedding
        return {"output":head_score, "image_feature":neck_feature, "text_feature":text_feature}