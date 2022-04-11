'''
Author: Thyssen Wen
Date: 2022-03-25 21:27:22
LastEditors: Thyssen Wen
LastEditTime: 2022-04-11 19:15:28
Description: model backbone script
FilePath: /ETESVS/model/backbone.py
'''
import torch
import torch.nn as nn

from .resnet import ResNet
from .resnet_tsm import ResNetTSM
from .resnet_tsm import TemporalShift
# form neckwork
# model_urls = {
#     "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
#     "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
#     "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
#     "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
#     "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth"
# }
# from local
model_urls = {
    18: "./data/resnet18-f37072fd.pth",
    34: "./data/resnet34-b627a593.pth",
    50: "./data/resnet50-0676ba61.pth",
    101: "./dataresnet101-63fe2227.pth",
    152: "./data/resnet152-394f9c45.pth"
}

class ETESVSBackBone(nn.Module):
    def __init__(self,
                 clip_seg_num=30,
                 shift_div=30,
                 name='ResNet',
                 pretrained=True,
                 depth=50):
        super().__init__()
        assert name in ['ResNet', 'ResNetTSM']

        assert depth in [18, 34, 50, 101, 152]

        if depth == 18 and pretrained == True:
            pretrained = model_urls[depth]
        elif depth == 34 and pretrained == True:
            pretrained = model_urls[depth]
        elif depth == 50 and pretrained == True:
            pretrained = model_urls[depth]
        elif depth == 101 and pretrained == True:
            pretrained = model_urls[depth]
        elif depth == 152 and pretrained == True:
            pretrained = model_urls[depth]

        if name == 'ResNet':
            self.model = ResNet(
                depth=depth,
                pretrained=pretrained
            )
        elif name == 'ResNetTSM':
            self.model = ResNetTSM(
                depth=depth,
                pretrained=pretrained,
                clip_seg_num=clip_seg_num,
                shift_div=shift_div
            )

        self.name = name
    
    def init_weights(self):
        self.model.init_weights()

    def _clear_memory_buffer(self):
        # self.apply(self._clean_activation_buffers)
        pass
    
    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), TemporalShift):
            m._reset_memory()
        
    def forward(self, x):
        if self.name in ['ResNet', 'ResNetTSM']:
            # x.shape=[N,T,C,H,W], for most commonly case
            # x [N, 1, T]
            x = torch.reshape(x, [-1] + list(x.shape[2:]))
            # x [N * T, C, H, W]
        return self.model(x)