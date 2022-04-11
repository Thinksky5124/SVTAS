'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-11 16:30:48
Description: model neck
FilePath: /ETESVS/model/neck.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import BasicBlock
from mmcv.cnn import ConvModule
from utils.logger import get_logger
from torchvision.utils import make_grid

class ETESVSNeck(nn.Module):
    def __init__(self,
                 num_classes=11,
                 num_layers=4,
                 out_channel=64,
                 in_channel=2048,
                 clip_seg_num=30,
                 drop_ratio=0.5,
                 sample_rate=4,
                 data_format="NCHW"):
        super().__init__()
        self.num_layers = num_layers
        self.out_channel = out_channel
        self.clip_seg_num = clip_seg_num
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.sample_rate = sample_rate

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"
        self.data_format = data_format
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.memory = MemoryMovingAverageConvModule()

        self.dropout = nn.Dropout(p=self.drop_ratio)
        
        self.fc = nn.Linear(self.in_channel, num_classes)

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.memory._reset_memory()
        # pass

    def forward(self, x, masks):
        # x.shape = [N * num_segs, 2048, 7, 7]
        x = self.avgpool(x)
        # x.shape = [N * num_segs, 2048, 1, 1]

        # segmentation feature branch
        # [N * num_segs, 2048]
        seg_x = torch.squeeze(x)
        # [N, num_segs, 2048]
        seg_feature = torch.reshape(seg_x, shape=[-1, self.clip_seg_num, seg_x.shape[-1]])

        # [N, 2048, num_segs]
        seg_feature = torch.permute(seg_feature, dims=[0, 2, 1])
        # [N, 2048, num_segs]
        seg_feature = self.memory(seg_feature, masks)

        # recognition branch
        cls_feature = torch.permute(seg_feature, dims=[0, 2, 1])
        cls_feature = torch.reshape(cls_feature, shape=[-1, self.in_channel]).unsqueeze(-1).unsqueeze(-1)
        if self.dropout is not None:
            x = self.dropout(cls_feature)  # [N * num_seg, in_channels, 1, 1]

        if self.data_format == 'NCHW':
            x = torch.reshape(x, x.shape[:2])
        else:
            x = torch.reshape(x, x.shape[::3])
        score = self.fc(x)  # [N * num_seg, num_class]
        score = torch.reshape(
            score, [-1, self.clip_seg_num, score.shape[1]])  # [N, num_seg, num_class]
        score = torch.mean(score, axis=1)  # [N, num_class]
        cls_score = torch.reshape(score,
                               shape=[-1, self.num_classes])  # [N, num_class]
        return seg_feature, cls_score

class MemoryMovingAverageConvModule(nn.Module):
    def __init__(self,
                 in_channels=2048,
                 out_channels=2048,
                 hidden_channels=512,
                 clip_seg_num=30):
        super().__init__()
        self.memory_matrix_dim = hidden_channels
        self.in_channel = in_channels
        self.clip_seg_num = clip_seg_num
        self.out_channels = out_channels
        conv_cfg = dict(type='Conv1d')
        norm_cfg = dict(type='BN1d', requires_grad=True)
        self.reduce_conv = BasicBlock(self.in_channel,
                                     self.memory_matrix_dim,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=None,
                                     downsample=ConvModule(
                                        self.in_channel,
                                        self.memory_matrix_dim,
                                        kernel_size=1,
                                        conv_cfg=conv_cfg,
                                        act_cfg=None))
        self.down_sample_conv_1 = nn.Sequential(
            nn.Conv1d(self.memory_matrix_dim, self.memory_matrix_dim, 3, 1, 1),
            nn.AvgPool1d(2, 2))
        self.down_sample_conv_2 = nn.Sequential(
            nn.Conv1d(self.memory_matrix_dim, self.memory_matrix_dim, 3, 1, 1),
            nn.AvgPool1d(2, 2),
            nn.ReLU())
        self.conv_out = ConvModule(
            self.memory_matrix_dim,
            self.out_channels,
            kernel_size=1,
            act_cfg=None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.avg_pool = nn.AdaptiveAvgPool1d(self.clip_seg_num * 2)

        self.register_buffer('memory_init', torch.Tensor(self.memory_matrix_dim, self.clip_seg_num * 2))
        self.register_buffer('last_x_init', torch.Tensor(self.memory_matrix_dim, self.clip_seg_num))

        # Initialize memory bias
        nn.init.constant_(self.memory_init, 0)
        nn.init.constant_(self.last_x_init, 0)
        self.memory_reset_flag = True

        self.tensorboard_writer = get_logger("ETESVS", tensorboard=True)
        self.step = 0

    def _reset_memory(self):
        self.memory = self.memory_init.clone()
        self.last_x = self.last_x_init.clone()
        self.memory_reset_flag = True

    def forward(self, x, masks):
        # x.shape[N, memory_matrix_dim, T]
        reduce_x = self.reduce_conv(x)
        if self.memory_reset_flag is True:
            self.memory = self.memory_init.repeat(reduce_x.shape[0], 1, 1).clone()
            self.last_x = self.last_x_init.repeat(reduce_x.shape[0], 1, 1).clone()
            self.memory_reset_flag = False
        
        moving_avg_feature = torch.cat([self.memory, self.last_x, reduce_x], dim=-1)
        out = self.down_sample_conv_1(moving_avg_feature)
        out = self.down_sample_conv_2(out)
        out = self.conv_out(out)
        out = (x + out) * masks[:, 0:1, :]

        self.memory = self.avg_pool(torch.cat([self.memory, self.last_x], dim=-1)).detach().clone()
        self.last_x = reduce_x.detach().clone()

        # self.tensorboard_writer.add_image('out', out.permute([0, 2, 1]).detach().cpu(), self.step)
        # self.tensorboard_writer.add_image('memory', self.memory.permute([0, 2, 1]).detach().cpu(), self.step)
        # self.tensorboard_writer.add_image('last_x', self.last_x.permute([0, 2, 1]).detach().cpu(), self.step)
        # self.step = self.step + 1

        return out