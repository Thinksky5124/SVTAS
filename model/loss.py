import torch
import torch.nn as nn

class ETETSLoss(nn.Module):
    def __init__(self,
                 seg_weight=1.0,
                 cls_weight=1.0,
                 ignore_index=-100):
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, mask):
        pass