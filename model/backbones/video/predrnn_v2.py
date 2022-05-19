'''
Author       : Thyssen Wen
Date         : 2022-05-16 14:00:56
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 21:34:28
Description  : PredRNN v2 model ref:https://github.com/thuml/predrnn-pytorch/blob/master/core/models/predrnn_v2.py
FilePath     : /ETESVS/model/backbones/video/predrnn_v2.py
'''
import torch
import torch.nn as nn
from ..utils.stlstm import SpatioTemporalLSTMCell
import torch.nn.functional as F
from utils.logger import get_logger
from mmcv.runner import load_checkpoint
from ...builder import BACKBONES

@BACKBONES.register()
class PredRNNV2(nn.Module):
    def __init__(self,
                 num_layers=4,
                 num_hidden=[128, 128, 128, 128],
                 img_width=224,
                 img_channel=3,
                 filter_size=5,
                 stride=1,
                 layer_norm=False,
                 pretrained=None):
        super(PredRNNV2, self).__init__()
        self.pretrained = pretrained
        self.frame_channel = img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        assert self.num_layers == len(self.num_hidden), "num_layers must mach length of num_hidden list"
        cell_list = []

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], img_width, filter_size,
                                       stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.init_hidden_state_falg = True

    def _clear_memory_buffer(self):
        self.init_hidden_state_falg = True
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)

    def forward(self, frames_tensor, mask_true):
        # [N C T H W] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 2, 1, 3, 4).contiguous()
        mask_true = mask_true.permute(0, 2, 1, 3, 4).contiguous()

        batch = frames.shape[0]
        temporal_len = frames.shape[1]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        if self.init_hidden_state_falg is True:
            self.h_t_list = []
            self.c_t_list = []

            for i in range(self.num_layers):
                zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(frames.device)
                self.h_t_list.append(zeros)
                self.c_t_list.append(zeros)

            self.memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(frames.device)
            self.init_hidden_state_falg = False
        
        h_t = self.h_t_list
        c_t = self.c_t_list
        memory = self.memory

        for t in range(temporal_len):
            frame = frames[:, t] * mask_true[:, t]
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](frame, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = h_t[self.num_layers - 1]
            next_frames.append(x_gen)
        
        # memory
        for l in range(self.num_layers):
            self.h_t_list[l] = h_t[l].detach().clone()
            self.c_t_list[l] = c_t[l].detach().clone()
            self.memory = memory.detach().clone()

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 2, 0, 3, 4).contiguous()
        return next_frames