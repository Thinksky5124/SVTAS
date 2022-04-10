'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-10 15:55:59
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
        self.memory = MemoryCache()

        self.dropout = nn.Dropout(p=self.drop_ratio)
        
        self.fc = nn.Linear(self.in_channel, num_classes)

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.memory._resert_memory()
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
        pre_seg_feature = torch.permute(seg_feature, dims=[0, 2, 1])
        # [N, 2048, num_segs]
        seg_feature = self.memory(pre_seg_feature, masks)

        # recognition branch
        cls_feature = torch.permute(pre_seg_feature, dims=[0, 2, 1])
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

class MemoryCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_matrix_dim = 512
        self.memory_len = 90
        self.in_channel = 2048
        conv_cfg = dict(type='Conv1d')
        norm_cfg = dict(type='BN1d', requires_grad=True)
        self.reduce_x = nn.Conv1d(self.in_channel, self.memory_matrix_dim, 1)
        self.up_x = nn.Conv1d(self.memory_matrix_dim, self.in_channel, 1)
        self.read_att = nn.MultiheadAttention(self.memory_matrix_dim, 1, batch_first=True)
        self.read_conv = ReadBasicBlock(self.memory_matrix_dim + self.in_channel,
                                    self.in_channel,
                                    dropout=0.5,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg)
        self.update_conv = BasicBlock(self.memory_matrix_dim,
                                     self.memory_matrix_dim,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg)
        self.avg_pool = nn.AdaptiveAvgPool1d(self.memory_len)

        self.memory = None

        # self.tensorboard_writer = get_logger("ETESVS", tensorboard=True)
        # self.step = 0

    def _resert_memory(self):
        self.memory = None

    def _write(self):
        # write
        # update = self.update_conv(self.last_x).transpose(1, 2)
        cat_video_feature = torch.concat([self.memory, self.last_x.transpose(1, 2)], dim=1)
        memory = self.avg_pool(cat_video_feature.transpose(1, 2)).transpose(1, 2)
        self.memory = memory.detach().clone()
        # self.tensorboard_writer.add_image('update', update.detach().cpu(), self.step)
        return memory

    def _read(self, re_x, x, memory, masks):
        # x.shape[N, T, in_channel]
        x_transpose = torch.permute(x, dims=[0, 2, 1])
        # self.tensorboard_writer.add_image('x', x.detach().cpu(), self.step)
        # read
        out, _ = self.read_att(re_x, memory, memory)
        
        # [N, 4096, num_segs]
        out_cat = torch.permute(torch.cat([x_transpose, out], dim=2), dims=[0, 2, 1])
        
        # self.tensorboard_writer.add_image('memory', memory.detach().cpu(), self.step)
        
        out = self.read_conv(out_cat, x) * masks[:, 0:1, :]
        # self.tensorboard_writer.add_image('out', out.permute([0, 2, 1]).detach().cpu(), self.step)
        # self.step = self.step + 1
        return out

    def forward(self, x, masks):
        # x.shape[N, memory_matrix_dim, T]
        re_x = self.reduce_x(x)
        if self.memory is None:
            self.memory = torch.zeros([x.shape[0], self.memory_len, self.memory_matrix_dim]).to(x.device)
            memory = self.memory.detach()
        else:
            memory = self._write()

        x_transpose = torch.permute(re_x, dims=[0, 2, 1])
        self.last_x = re_x.detach().clone()
        out = self._read(x_transpose, x, memory, masks)
        return out

class ReadBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, dropout, conv_cfg, norm_cfg):
        super().__init__(inplanes, planes, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, read_cat_x, raw_x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        identity = raw_x

        out = self.conv1(read_cat_x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(raw_x)

        out = self.dropout(out)
        out = out + identity
        out = self.relu(out)

        return out