'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-04-09 16:41:01
Description: model neck
FilePath: /ETESVS/model/neck.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.tensorboard_writer = get_logger("ETESVS", tensorboard=True)
        self.step = 0

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
        pass

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
        self.tensorboard_writer.add_image('memory', make_grid(
            [pre_seg_feature[0].detach().cpu(),
            seg_feature[0].detach().cpu()]), self.step)
        self.step = self.step + 1

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

class MemoryCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_matrix_dim = 2048
        self.memory_len = 30
        self.memory_hidden_dim = 2048
        self.in_channel = 2048
        self.read_att = nn.MultiheadAttention(self.memory_hidden_dim, 1, batch_first=True)
        self.add_att = nn.MultiheadAttention(self.memory_hidden_dim, 1, batch_first=True)
        self.erase_att = nn.MultiheadAttention(self.memory_hidden_dim, 1, batch_first=True)
        self.read_conv = nn.Conv1d(4096, self.in_channel, 1)
        self.memory = None
    
    def _resert_memory(self):
        self.memory = None

    def forward(self, x, masks):
        # x.shape[N, memory_matrix_dim, T]
        
        if self.memory is None:
            self.memory = torch.zeros([x.shape[0], self.memory_len, self.memory_matrix_dim], requires_grad=False).to(x.device)
        else:
            # write
            add, _ = self.add_att(self.last_x, self.memory, self.last_x)
            erase, _ = self.erase_att(self.last_x, self.memory, self.memory)
            erase = self.memory - erase
            self.memory = (erase + add).detach().clone()

        # x.shape[N, T, in_channel]
        x = torch.permute(x, dims=[0, 2, 1])
        # read
        out, _ = self.read_att(x, self.memory, self.memory)
        self.last_x = x.detach().clone()
        # [N, 4096, num_segs]
        out_cat = torch.permute(torch.cat([x, out], dim=2), dims=[0, 2, 1])
        
        return self.read_conv(out_cat) * masks[:, 0:1, :]
        
# class MemoryCache(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.memory_matrix_dim = 2048
#         self.in_channel = 2048
#         self.erase_conv = nn.Sequential(nn.Conv1d(self.in_channel, self.memory_matrix_dim, 1), nn.ReLU())
#         self.add_conv = nn.Sequential(nn.Conv1d(self.in_channel, self.memory_matrix_dim, 1), nn.ReLU())
#         self.read_conv = nn.Sequential(nn.Conv1d(self.in_channel, self.memory_matrix_dim, 1), nn.ReLU())
#         self.memory = None
#         self.last_att_map = None
#         # The memory bias allows the heads to learn how to initially address
#         # memory locations by content
#         self.register_buffer('mem_bias', torch.Tensor(self.memory_matrix_dim, self.memory_matrix_dim))

#         # Initialize memory bias
#         nn.init.kaiming_normal_(self.mem_bias)
    
#     def _resert_memory(self):
#         self.memory = None
#         self.last_att_map = None

#     def forward(self, x, masks):
#         # x.shape[N, memory_matrix_dim, T]
        
#         if self.memory is None:
#             self.memory = self.mem_bias.clone().repeat(x.shape[0], 1, 1).to(x.device)
#         else:
#             # first write
#             # erase.shape[N, memory_matrix_dim, T]
#             erase = self.erase_conv(x) * masks[:, 0:1, :]
#             # add.shape[N, memory_matrix_dim, T]
#             add = self.add_conv(x) * masks[:, 0:1, :]
#             erase_transpose = torch.permute(erase, dims=[0, 2, 1])
#             add_transpose = torch.permute(add, dims=[0, 2, 1])
#             att_map_transpose = torch.permute(self.last_att_map, dims=[0, 2, 1])

#             self.memory = (self.memory * (1 - torch.bmm(att_map_transpose, erase_transpose)) + torch.bmm(att_map_transpose, add_transpose)).detach().clone()

#         # second read
#         # generate att matrix
#         x = self.read_conv(x) * masks[:, 0:1, :]
#         x_transpose = torch.permute(x, dims=[0, 2, 1])
#         att_map = torch.bmm(x_transpose, self.memory) / math.sqrt(self.memory_matrix_dim)
#         read_cache = torch.bmm(att_map, self.memory).detach()
#         read_cache_transpose = torch.permute(read_cache, dims=[0, 2, 1])
#         # out.shape[N, memory_matrix_dim * 2, T]
#         out = torch.cat([x, read_cache_transpose], dim=1)

#         # memory att map
#         self.last_att_map = att_map.detach().clone()

#         return out
