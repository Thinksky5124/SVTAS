import torch
import torch.nn as nn

class ETETSNeck(nn.Module):
    def __init__(self,
                 buffer_channels=512,
                 hidden_channels=64,
                 num_layers=5,
                 clip_seg_num=15,
                 clip_buffer_num=0,
                 sliding_strike=5):
        super().__init__()
        self.buffer_channels = buffer_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.clip_seg_num = clip_seg_num
        self.clip_buffer_num = clip_buffer_num
        self.sliding_strike = sliding_strike
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, seg_mask, memery_buffer, mask_buffer):
        # x.shape = [N * num_segs, 2048, 7, 7]
        x = self.avgpool(x)
        # x.shape = [N * num_segs, 2048, 1, 1]
        cls_feature = x

        # segmentation branch
        # [N * num_segs, 2048]
        seg_x = torch.squeeze(x)
        # [N, num_segs, 2048]
        seg_feature = torch.reshape(seg_x, shape=[-1, self.clip_seg_num, seg_x.shape[-1]])  
        # [N, 2048, num_segs]
        seg_feature = torch.permute(seg_feature, dims=[0, 2, 1])

        return seg_feature, cls_feature, memery_buffer, mask_buffer