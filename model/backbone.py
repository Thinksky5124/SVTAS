import torch
import torch.nn as nn

from .resnet import ResNet
from .resnet_tsm import ResNetTSM

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

class ETETSBackBone(nn.Module):
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
    
    def init_weights(self):
        self.backbone.init_weights()
    
    def forward(self, x):
        return self.backbone(x)