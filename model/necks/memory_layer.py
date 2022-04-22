'''
Author: Thyssen Wen
Date: 2022-04-13 14:01:12
LastEditors: Thyssen Wen
LastEditTime: 2022-04-22 21:24:57
Description: file content
FilePath: /ETESVS/model/necks/memory_layer.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from .convlstm import ConvLSTM
from utils.logger import get_logger

class LSTMResidualLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=1, dropout=0.5, bidirectional=False):
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
        self.dropout = dropout
        self.encoder = ConvLSTM(in_channels, hidden_channels, (3, 3), num_layers, batch_first=True)
        # self.decoder = nn.LSTM(in_channels, hidden_channels, num_layers, batch_first=True)
        self.de_dropout = nn.Dropout(p=self.dropout)
        self.en_dropout = nn.Dropout(p=self.dropout)
        self.input_dropout = nn.Dropout(p=self.dropout)
        self.fc_frames_cls = nn.Linear(hidden_channels, num_classes)

        self.writer = get_logger(name="ETESVS", tensorboard=True)
        self.step = 0

        # init buffer
        self.hidden = None
        self.cell = None
        self.last_x = None
    
    def _reset_memory(self):
        self.hidden = None
        self.cell = None
        self.last_x = None

    def forward(self, x, masks):
        # if self.hidden is None:
        # # if self.hidden is None or self.cell is None:
        #     self.hidden = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels).to(x.device)
        #     self.cell = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels).to(x.device)
        #     self.last_x = torch.zeros(x.shape[0], 1, self.in_channels).to(x.device)
        
        # [N T C H W]
        # memory encoder
        seg_feature, (en_hidden, en_cell) = self.encoder(x)

        # if self.en_dropout is not None:
        #     frames_feature = self.en_dropout(frames_feature)  # [N * num_seg, in_channels]
        # frames_feature = x + frames_feature

        # memory decoder predict
        # hidden = self.hidden
        # cell = self.cell
        # seg_feature_list = []
        # for frame in range(x.shape[1]):
        #     if frame <= 0:
        #         frame_feature =  self.last_x + x[:, frame, :].unsqueeze(1).detach().clone()
        #     else:
        #         frame_feature =  frame_feature + x[:, frame, :].unsqueeze(1).detach().clone()
        #     frame_feature, (hidden, cell) = self.decoder(frame_feature, (hidden, cell))
        #     seg_feature_list.append(frame_feature)
        # seg_feature = torch.cat(seg_feature_list, dim=1)
        # memory
        self.last_x = x[:, -1:, :].detach().clone()
        self.hidden = en_hidden.detach().clone()
        self.cell = en_cell.detach().clone()
        
        if self.de_dropout is not None:
            seg_feature = self.de_dropout(seg_feature)  # [N * num_seg, in_channels]
        # seg_feature = seg_feature + seg_feature_dropout

        cls_feature = torch.reshape(seg_feature, shape=[-1, self.hidden_channels])
        frames_score = self.fc_frames_cls(cls_feature)
        # [N, num_seg, num_class]
        frames_score = torch.reshape(
            frames_score, [seg_feature.shape[0], -1, self.num_classes]) * masks.transpose(1, 2)

        if self.input_dropout is not None:
            x = self.input_dropout(x)
        neck_feature = x + seg_feature
        
        # self.writer.add_image("x", x.detach().cpu(), self.step)
        # self.writer.add_image("seg_feature", seg_feature.detach().cpu(), self.step)
        # self.writer.add_image("neck_feature", neck_feature.detach().cpu(), self.step)
        # self.writer.add_image("en_hidden", en_hidden.detach().cpu(), self.step)
        # self.writer.add_image("en_cell", en_cell.detach().cpu(), self.step)
        # self.step = self.step + 1

        # [N , in_channels, num_seg]
        neck_feature = torch.permute(neck_feature, dims=[0, 2, 1])

        return neck_feature * masks[:, 0:1, :], frames_score
        
# class LSTMResidualLayer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_classes, num_layers=1, dropout=0.5, bidirectional=False):
#         super().__init__()
#         self.hidden_channels = hidden_channels
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.sample_rate = 4
#         if bidirectional is False:
#             self.direction = 1
#         else:
#             self.direction = 2
#         self.dropout = dropout
#         self.encoder = nn.LSTM(in_channels, hidden_channels, num_layers, bidirectional=False, batch_first=True)
#         self.decoder = nn.LSTM(in_channels, hidden_channels, num_layers, bidirectional=False, batch_first=True)
#         self.de_dropout = nn.Dropout(p=self.dropout)
#         self.en_dropout = nn.Dropout(p=self.dropout)
#         self.input_dropout = nn.Dropout(p=self.dropout)
#         self.fc_frames_cls = nn.Linear(hidden_channels, num_classes)

#         self.writer = get_logger(name="ETESVS", tensorboard=True)
#         self.step = 0

#         # init buffer
#         self.hidden = None
#         self.cell = None
#         self.last_x = None
    
#     def _reset_memory(self):
#         self.hidden = None
#         self.cell = None
#         self.last_x = None

#     def forward(self, x, masks):
#         if self.hidden is None:
#         # if self.hidden is None or self.cell is None:
#             self.hidden = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels).to(x.device)
#             self.cell = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels).to(x.device)
#             self.last_x = torch.zeros(x.shape[0], 1, self.in_channels).to(x.device)
        
#         # [N T D]
#         x = x.transpose(1, 2)
#         # memory encoder
#         seg_feature, (en_hidden, en_cell) = self.encoder(x, (self.hidden, self.cell))

#         # if self.en_dropout is not None:
#         #     frames_feature = self.en_dropout(frames_feature)  # [N * num_seg, in_channels]
#         # frames_feature = x + frames_feature

#         # memory decoder predict
#         # hidden = self.hidden
#         # cell = self.cell
#         # seg_feature_list = []
#         # for frame in range(x.shape[1]):
#         #     if frame <= 0:
#         #         frame_feature =  self.last_x + x[:, frame, :].unsqueeze(1).detach().clone()
#         #     else:
#         #         frame_feature =  frame_feature + x[:, frame, :].unsqueeze(1).detach().clone()
#         #     frame_feature, (hidden, cell) = self.decoder(frame_feature, (hidden, cell))
#         #     seg_feature_list.append(frame_feature)
#         # seg_feature = torch.cat(seg_feature_list, dim=1)
#         # memory
#         self.last_x = x[:, -1:, :].detach().clone()
#         self.hidden = en_hidden.detach().clone()
#         self.cell = en_cell.detach().clone()
        
#         if self.de_dropout is not None:
#             seg_feature = self.de_dropout(seg_feature)  # [N * num_seg, in_channels]
#         # seg_feature = seg_feature + seg_feature_dropout

#         cls_feature = torch.reshape(seg_feature, shape=[-1, self.hidden_channels])
#         frames_score = self.fc_frames_cls(cls_feature)
#         # [N, num_seg, num_class]
#         frames_score = torch.reshape(
#             frames_score, [seg_feature.shape[0], -1, self.num_classes]) * masks.transpose(1, 2)

#         if self.input_dropout is not None:
#             x = self.input_dropout(x)
#         neck_feature = x + seg_feature
        
#         # self.writer.add_image("x", x.detach().cpu(), self.step)
#         # self.writer.add_image("seg_feature", seg_feature.detach().cpu(), self.step)
#         # self.writer.add_image("neck_feature", neck_feature.detach().cpu(), self.step)
#         # self.writer.add_image("en_hidden", en_hidden.detach().cpu(), self.step)
#         # self.writer.add_image("en_cell", en_cell.detach().cpu(), self.step)
#         # self.step = self.step + 1

#         # [N , in_channels, num_seg]
#         neck_feature = torch.permute(neck_feature, dims=[0, 2, 1])

#         return neck_feature * masks[:, 0:1, :], frames_score
    