import torch
import torch.nn as nn

from .backbone import resnet18, resnet34, resnet50, resnet101
from .neck import ETETSNeck
from .head import ETETSHead

class ETETS(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        if backbone.depth == 18:
            self.backbone = resnet18(backbone.pretrained)
        elif backbone.depth == 34:
            self.backbone = resnet34(backbone.pretrained)
        elif backbone.depth == 50:
            self.backbone = resnet50(backbone.pretrained)
        elif backbone.depth == 101:
            self.backbone = resnet101(backbone.pretrained)
        else:
            self.backbone = None
        self.neck = ETETSNeck(**neck)
        self.head = ETETSHead(**head)

    def forward(self, x):
        pass