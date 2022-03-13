import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

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
                 data_format="NCHW"):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps
        self.cls_in_channels = cls_in_channels
        self.seg_in_channels = seg_in_channels
        self.sample_rate = sample_rate
        self.drop_ratio = drop_ratio

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"

        self.data_format = data_format

        self.dropout = nn.Dropout(p=self.drop_ratio)
        self.seg_conv = SingleStageModel(num_layers, num_f_maps, seg_in_channels,
                                       num_classes)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(num_layers, num_f_maps, num_classes,
                                 num_classes)) for s in range(num_stages - 1)
        ])
        
        self.fc = nn.Linear(self.cls_in_channels, num_classes)

    def forward(self, seg_feature, cls_feature, masks):
        # segmentation branch
        # seg_feature [N, in_channels, temporal_len]
        # Interploate upsample
        seg_x_upsample = F.interpolate(
            input=seg_feature,
            scale_factor=[self.sample_rate],
            mode="nearest")

        out = self.seg_conv(seg_x_upsample, masks)
        outputs = out.unsqueeze(0)
        # seg_feature [stage_num, N, num_class, temporal_len]
        for s in self.stages:
            out = s(F.softmax(out, dim=1), masks)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        seg_score = outputs

        if self.dropout is not None:
            x = self.dropout(cls_feature)  # [N * num_seg, in_channels, 1, 1]

        if self.data_format == 'NCHW':
            x = torch.reshape(x, x.shape[:2])
        else:
            x = torch.reshape(x, x.shape[::3])
        score = self.fc(x)  # [N * num_seg, num_class]
        cls_score = torch.reshape(
            score, [-1, seg_feature.shape[2], score.shape[1]])  # [N, num_seg, num_class]
        # score = torch.mean(score, axis=1)  # [N, num_class]
        # cls_score = torch.reshape(score,
        #                        shape=[-1, self.num_classes])  # [N, num_class]
        return seg_score, cls_score

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]