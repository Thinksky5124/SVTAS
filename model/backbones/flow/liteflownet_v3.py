'''
Author       : Thyssen Wen
Date         : 2022-05-18 21:29:02
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-19 18:38:56
Description  : LiteFlowNet V3 model ref:https://github.com/lhao0301/pytorch-liteflownet3/blob/main/run.py
FilePath     : /ETESVS/model/backbones/flow/liteflownet_v3.py
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from ..utils.liteflownet_v3 import Features, Matching, Regularization, Subpixel, BackWarp

from utils.logger import get_logger
from ...builder import BACKBONES


@BACKBONES.register()
class LiteFlowNetV3(nn.Module):
    def __init__(self,
                 pretrained=None,
                 freeze=True,
                 extract_mode=True,
                 scale_factor_flow=20.0):
        super(LiteFlowNetV3, self).__init__()
        self.pretrained = pretrained
        self.scale_factor_flow = scale_factor_flow
        self.freeze = freeze
        self.extract_mode = extract_mode

        # memory
        self.memory_frame = None

        self.backwarp = BackWarp()
        self.netFeatures = Features()
        self.netMatching = nn.ModuleList([Matching(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
        self.netSubpixel = nn.ModuleList([Subpixel(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
        self.netRegularization = nn.ModuleList([Regularization(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])

    def _clear_memory_buffer(self):
        self.memory_frame = None
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)
                if self.freeze is True:
                    self.eval()
                    for param in self.parameters():
                        param.requires_grad = False
            else:
                raise TypeError('pretrained must be a str or None')
    
    def pre_precessing(self, x):
        # x.shape [N, T, C, H, W]
        if self.memory_frame is None:
            # img1.shape [N, (T - 1), C, H, W]
            img1 = x[:, :-1]
            img1 = torch.reshape(img1, shape=[-1] + list(img1.shape[-3:]))
            # img2.shape [N, (T - 1), C, H, W]
            img2 = x[:, 1:]
            img2 = torch.reshape(img2, shape=[-1] + list(img2.shape[-3:]))
            self.memory_frame = x[:, -2:-1].detach().clone()
        else:
            # img1.shape [N, T, C, H, W]
            img1 = torch.cat([self.memory_frame, x[:, :-1]], dim=1)
            img1 = torch.reshape(img1, shape=[-1] + list(img1.shape[-3:]))
            # img2.shape [N, T, C, H, W]
            img2 = x
            img2 = torch.reshape(img2, shape=[-1] + list(img2.shape[-3:]))
            self.memory_frame = x[:, -2:-1].detach().clone()

        intWidth = img1.shape[-1]
        intHeight = img2.shape[-2]

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        img1 = nn.functional.interpolate(input=img1, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        img2 = nn.functional.interpolate(input=img2, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        temporal_len = img1.shape[0]
        orig_size = (int(intHeight), int(intWidth))
        input_size = (int(intPreprocessedHeight), int(intPreprocessedWidth))
        
        return img1, img2, input_size, orig_size, temporal_len
    
    def post_precessing(self, flow, input_size, orig_size, temporal_len):
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False) * self.scale_factor_flow
        flow[:, 0, :, :] *= float(orig_size[1]) / float(input_size[1])
        flow[:, 1, :, :] *= float(orig_size[0]) / float(input_size[0])

        # [N, T, C, H, W]
        refine_flow = torch.reshape(flow, shape=[-1] + [temporal_len, 2] + list(flow.shape[-2:]))

        return refine_flow

    def forward(self, image1, image2):
        
        image1 = image1 - torch.mean(image1, (2, 3), keepdim=True)
        image2 = image2 - torch.mean(image2, (2, 3), keepdim=True)

        tenFeaturesFirst = self.netFeatures(image1)
        tenFeaturesSecond = self.netFeatures(image2)

        image1 = [ image1 ]
        image2 = [ image2 ]

        for intLevel in [ 2, 3, 4, 5 ]:
            image1.append(nn.functional.interpolate(input=image1[-1], size=(tenFeaturesFirst[intLevel].shape[2], tenFeaturesFirst[intLevel].shape[3]), mode='bilinear', align_corners=False))
            image2.append(nn.functional.interpolate(input=image2[-1], size=(tenFeaturesSecond[intLevel].shape[2], tenFeaturesSecond[intLevel].shape[3]), mode='bilinear', align_corners=False))

        tenFlow = None
        tenConf = None

        for intLevel in [ -1, -2, -3, -4 ]:
            tenFlow, tenConf = self.netMatching[intLevel](image1[intLevel], image2[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow, tenConf, self.backwarp)
            tenFlow = self.netSubpixel[intLevel](image1[intLevel], image2[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow, self.backwarp)
            tenFlow, tenConf = self.netRegularization[intLevel](image1[intLevel], image2[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow, self.backwarp)

        if self.training:
            return tenFlow
        else:
            return tenFlow