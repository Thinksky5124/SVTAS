import torch
import torch.nn as nn

class ETETSHead(nn.Module):
    def __init__(self,
                 num_classes=48,
                 num_stages=1,
                 num_layers=4,
                 num_f_maps=64,
                 cls_in_channels=2048,
                 seg_in_channels=2048,
                 sample_rate=4,
                 drop_ratio=0.5,
                 std=0.001):
        super().__init__()
        self.cls_in_channels = cls_in_channels
        self.fc = nn.Linear(self.cls_in_channels, num_classes)

    def forward(self, x):
        pass