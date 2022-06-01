'''
Author       : Thyssen Wen
Date         : 2022-05-04 14:57:21
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 19:45:37
Description  : file content
FilePath     : /ETESVS/model/architectures/optical_flow_estimator.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import build_backbone

from ...builder import ARCHITECTURE

@ARCHITECTURE.register()
class OpticalFlowEstimation(nn.Module):
    def __init__(self,
                 model=None,
                 loss=None):
        super().__init__()
        if model is not None:
            self.model = build_backbone(model)
        else:
            self.model = None

        self.init_weights()

    def init_weights(self):
        if self.model is not None:
            self.model.init_weights(child_model=False)
    
    def _clear_memory_buffer(self):
        self.model._clear_memory_buffer()

    def forward(self, input_data):
        flow_imgs = input_data['imgs']

        # feature.shape=[N,T,C,H,W], for most commonly case
        if self.model.extract_mode is True:
            img1, img2, input_size, orig_size, temporal_len = self.model.pre_precessing(flow_imgs)
        else:
            img1 = flow_imgs[:, :3, :, :]
            img2 = flow_imgs[:, 3:6, :, :]
            
        if self.model is not None:
            flow2 = self.model(img1, img2)
        else:
            flows = flow_imgs
        
        if not self.training and self.model.extract_mode is True:
            flows = self.model.post_precessing(flow2, input_size, orig_size, temporal_len)
        else:
            flows = flow2
        # seg_score [N,T,C,H,W]
        return flows