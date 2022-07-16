'''
Author       : Thyssen Wen
Date         : 2022-06-18 12:11:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-16 09:57:43
Description  : file content
FilePath     : /ETESVS/model/necks/memory_layer.py
'''
'''
Author: Thyssen Wen
Date: 2022-04-13 14:01:12
LastEditors: Thyssen Wen
LastEditTime: 2022-04-28 14:46:54
Description: file content
FilePath: /ETESVS/model/necks/memory_layer.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .convlstm import ConvLSTM
from utils.logger import get_logger

class ConvLSTMResidualLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.sample_rate = 4
        if bidirectional is False:
            self.direction = 1
        else:
            self.direction = 2
        
        self.conv_lstm = ConvLSTM(in_channels, hidden_channels, (3, 3), num_layers, batch_first=True)
        self.fc_frames_cls = nn.Linear(hidden_channels, num_classes)

        # self.writer = get_logger(name="SVTAS", tensorboard=True)
        # self.step = 0

        # init buffer
        self.hidden_state = None
    
    def _reset_memory(self):
        self.hidden_state = None

    def forward(self, x, masks):
        if self.hidden_state is None:
            hidden = torch.zeros_like(x[:, 0, :, :, :]).to(x.device)
            cell = torch.zeros_like(x[:, 0, :, :, :]).to(x.device)
            self.hidden_state = [(hidden.detach().clone(), cell.detach().clone()) for l in range(self.num_layers)]

        # [N T C H W]
        # memory encoder
        layer_output_list, last_state_list = self.conv_lstm(x, self.hidden_state)

        # memory
        self.hidden_state = [[last_state_list[l][i].detach().clone() for i in range(len(last_state_list[l]))] for l in range(self.num_layers)]
        
        neck_feature = layer_output_list[-1]
        
        # self.writer.add_image("x", x.detach().cpu(), self.step)
        # self.writer.add_image("seg_feature", seg_feature.detach().cpu(), self.step)
        # self.writer.add_image("neck_feature", neck_feature.detach().cpu(), self.step)
        # self.writer.add_image("en_hidden", en_hidden.detach().cpu(), self.step)
        # self.writer.add_image("en_cell", en_cell.detach().cpu(), self.step)
        # self.step = self.step + 1

        # [N , in_channels, num_seg]
        mask_change = masks[:, 0:1, :].transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return neck_feature * mask_change
    