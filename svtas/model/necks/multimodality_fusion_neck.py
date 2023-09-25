'''
Author       : Thyssen Wen
Date         : 2022-11-05 20:48:29
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-15 16:19:23
Description  : MultiModality Fusion Neck Model
FilePath     : /SVTAS/svtas/model/necks/multimodality_fusion_neck.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import build_neck
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class MultiModalityFusionNeck(nn.Module):
    def __init__(self,
                 clip_seg_num=32,
                 fusion_mode='stack',
                 fusion_neck_module=None) -> None:
        super().__init__()
        assert fusion_mode in ['stack']

        self.clip_seg_num = clip_seg_num
        self.fusion_mode = fusion_mode
        if fusion_neck_module is not None:
            self.fusion_neck_module = build_neck(fusion_neck_module)
        else:
            self.fusion_neck_module = None
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
    
    def _stack_feature(self, fusion_feature_dict):
        feature_list = []
        for k, v in fusion_feature_dict.items():
            feature_list.append(v)
        return torch.concat(feature_list, dim=1)

    def forward(self, fusion_feature_dict, masks):
        if self.fusion_mode == 'stack':
            feature = self._stack_feature(fusion_feature_dict)
        else:
            raise NotImplementedError
        masks = F.adaptive_max_pool1d(masks.squeeze(1), self.clip_seg_num, return_indices=False).unsqueeze(1)
        if self.fusion_neck_module is not None:
            return self.fusion_neck_module(feature, masks)
        else:
            return feature * masks