'''
Author       : Thyssen Wen
Date         : 2022-10-17 13:15:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 13:06:46
Description  : 3D TCN model
FilePath     : /SVTAS/svtas/model/heads/segmentation/tcn_3d_head.py
'''

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from svtas.model_pipline.torch_utils import constant_init, kaiming_init, xavier_init

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class TCN3DHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_stages=1,
                 num_layers=4,
                 sample_rate=4,
                 seg_in_channels=2048,
                 num_f_maps=64,
                 out_feature=False):
        super(TCN3DHead, self).__init__()
        self.seg_in_channels = seg_in_channels
        self.out_feature = out_feature
        self.num_f_maps = num_f_maps
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.num_layers = num_layers

        self.stage1 = SingleStage3DModel(num_layers, num_f_maps, seg_in_channels, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStage3DModel(num_layers, num_f_maps, num_f_maps, num_classes)) for s in range(num_stages-1)])

    def init_weights(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         xavier_init(m)
        #     elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
        #         constant_init(m, 1)

    def _clear_memory_buffer(self):
        # self.seg_conv._clear_memory_buffer()
        pass

    def forward(self, seg_feature, mask):
        # segmentation branch
        # seg_feature [N, seg_in_channels, num_segs, 7, 7]
        
        mask = mask[:, :, ::self.sample_rate]
        
        feature, out = self.stage1(seg_feature, mask)
        
        outputs = out.unsqueeze(0)
        for s in self.stages:
            feature, out = s(feature, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
            
        if self.out_feature is True:
            return feature * mask[:, 0:1, :].unsqueeze(-1).unsqueeze(-1), outputs
        return outputs

class SingleStage3DModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, output_channels):
        super(SingleStage3DModel, self).__init__()
        self.conv_1x1 = nn.Conv3d(dim, num_f_maps, (1, 1, 1))
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidual3DLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_out = nn.Conv1d(num_f_maps, output_channels, 1)

    def forward(self, x, mask):
        feature_embedding = self.conv_1x1(x)
        feature = feature_embedding
        for layer in self.layers:
            feature = layer(feature, mask)

        # [N C T H W]
        out = torch.permute(feature, dims=[0, 2, 1, 3, 4])
        # seg_feature [N, num_segs, C, 7, 7]
        out = torch.reshape(out, shape=[-1] + list(out.shape[-3:]))
        # seg_feature_pool.shape = [N * num_segs, C, 1, 1]
        out = self.avgpool(out)

        # seg_feature_pool.shape = [N, num_segs, 1280, 1, 1]
        out = torch.reshape(out, shape=[-1, feature.shape[2]] + list(out.shape[-3:]))

        # segmentation feature branch
        # [N, C, num_segs]
        out = out.squeeze(-1).squeeze(-1).transpose(1, 2)

        out = self.conv_out(out) * mask[:, 0:1, :]

        return feature * mask[:, 0:1, :].unsqueeze(-1).unsqueeze(-1), out


class DilatedResidual3DLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidual3DLayer, self).__init__()
        self.conv_dilated = nn.Conv3d(in_channels, out_channels, (3,3,3), stride=(1, 1, 1), padding=(dilation, dilation, dilation), dilation=(dilation, dilation, dilation))
        self.conv_1x1 = nn.Conv3d(out_channels, out_channels, (1, 1, 1))
        # self.norm = nn.LazyBatchNorm3d()
        self.norm = nn.Dropout()

    def forward(self, x, mask):
        # !
        # out = F.leaky_relu(self.conv_dilated(x))
        # !
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :].unsqueeze(-1).unsqueeze(-1)