import torch
import torch.nn as nn

class ETETSNeck(nn.Module):
    def __init__(self,
                 buffer_channels=512,
                 hidden_channels=64,
                 num_layers=5,
                 num_segs=15,
                 clip_buffer_num=0,
                 sliding_strike=5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        pass