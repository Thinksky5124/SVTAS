'''
Author       : Thyssen Wen
Date         : 2022-05-04 14:57:21
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 14:53:48
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/optical_flow/optical_flow_estimator.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..general import SeriousModel
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('architecture')
class OpticalFlowEstimation(SeriousModel):
    def __init__(self,
                 model=None,
                 weight_init_cfg = dict(
                     model=dict(child_model=False))):
        super().__init__(weight_init_cfg=weight_init_cfg, model=model)

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
        return {"output":flows}