import torch
import torch.nn as nn

from .resnet import ResNet
from .resnet_tsm import ResNetTSM

class ETETSBackBone(nn.Module):
    def __init__(self,
                 clip_seg_num=30,
                 shift_div=30,
                 name='ResNet',
                 pretrained=True,
                 depth=50):
        super().__init__()
        assert name in ['ResNet', 'ResNetTSM']

        if name == 'ResNet':
            self.backbone = ResNet(
                depth=depth,
                pretrained=pretrained
            )
        elif name == 'ResNetTSM':
            self.backbone = ResNetTSM(
                depth=depth,
                pretrained=pretrained,
                clip_seg_num=clip_seg_num,
                shift_div=shift_div
            )
    
    def forward(self, x):
        return self.backbone(x)