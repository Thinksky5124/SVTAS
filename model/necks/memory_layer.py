'''
Author: Thyssen Wen
Date: 2022-04-13 14:01:12
LastEditors: Thyssen Wen
LastEditTime: 2022-04-23 14:28:46
Description: file content
FilePath: /ETESVS/model/necks/memory_layer.py
'''
from multiprocessing import pool
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .convlstm import ConvLSTM
from utils.logger import get_logger

class CausalResidualLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, dilation, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.causal_conv = ConvModule(in_channels=in_channels,
                                      out_channels=hidden_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)
                                      
    def forward(self, x, masks):
        out = self.causal_conv(x)
        avg_x = F.adaptive_avg_pool1d(x, output_size=out.shape[-1])
        pool_mask = F.adaptive_max_pool1d(masks, output_size=out.shape[-1])
        return (avg_x + out) * pool_mask[:, 0:1, :]

class CausalTCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.num_layers = num_layers
        conv_cfg = dict(type="Conv1d")
        norm_cfg = dict(type="BN1d")
        act_cfg = dict(type='ReLU')
        self.embeding_conv = ConvModule(in_channels=in_channels,
                                        out_channels=hidden_channels,
                                        kernel_size=1,
                                        stride=1,
                                        dilation=1,
                                        conv_cfg=conv_cfg,
                                        norm_cfg=None,
                                        act_cfg=None)
        self.causal_conv_list = nn.ModuleList([copy.deepcopy(
            CausalResidualLayer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                kernel_size=2,
                stride=1,
                dilation=2 ** i,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )) for i in range(num_layers)])
        self.causal_out_conv = CausalResidualLayer(in_channels=in_channels,
                                                   hidden_channels=hidden_channels,
                                                   kernel_size=2,
                                                   stride=1,
                                                   dilation=1,
                                                   conv_cfg=conv_cfg,
                                                   norm_cfg=norm_cfg,
                                                   act_cfg=act_cfg)
        self.out_conv = ConvModule(in_channels=in_channels,
                                   out_channels=hidden_channels,
                                   kernel_size=1,
                                   stride=1,
                                   dilation=1,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=None,
                                   act_cfg=dict(type='ReLU'))

    def forward(self, x, masks):
        x = self.embeding_conv(x)
        out = x
        for causal_conv in self.causal_conv_list:
            out = causal_conv(out, masks)
        out = self.causal_out_conv(out, masks)
        out = self.out_conv(out) * masks[:, :, -out.shape[-1]:]
        return out

class CausalConvEncoderDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.num_layers = num_layers

        self.encoder_causal = CausalTCNBlock(in_channels=in_channels,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers)

        self.decoder_causal = CausalTCNBlock(in_channels=in_channels,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers)
        
        self.memory = None
        self.last_masks = None
    
    def _reset_memory(self):
        self.memory = None
        self.last_masks = None

    def forward(self, x, masks):
        if self.memory is None or self.last_masks is None:
            self.memory = torch.zeros_like(x).to(x.device)
            self.last_masks = torch.zeros_like(masks).to(masks.device)

        # encoder
        input_x = torch.concat([self.memory, x], dim=-1)
        input_masks = torch.concat([self.last_masks, masks], dim=-1)
        en_feature = self.encoder_causal(input_x, input_masks)

        # memory
        self.memory = en_feature.detach().clone()
        self.last_masks = masks.detach().clone()

        # decoder
        input_x = torch.concat([en_feature, x], dim=-1)
        de_feature = self.decoder_causal(input_x, input_masks)

        return de_feature

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
#         self.encoder = ConvLSTM(in_channels, hidden_channels, (3, 3), num_layers, batch_first=True)
#         self.decoder = ConvLSTM(in_channels, hidden_channels, (3, 3), num_layers, batch_first=True)
#         self.de_dropout = nn.Dropout(p=self.dropout)
#         self.en_dropout = nn.Dropout(p=self.dropout)
#         self.input_dropout = nn.Dropout(p=self.dropout)
#         self.fc_frames_cls = nn.Linear(hidden_channels, num_classes)

#         self.writer = get_logger(name="ETESVS", tensorboard=True)
#         self.step = 0

#         # init buffer
#         self.hidden_state = None
#         self.last_x = None
    
#     def _reset_memory(self):
#         self.hidden_state = None
#         self.last_x = None

#     def forward(self, x, masks):
#         if self.hidden_state is None:
#             hidden = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels, ).to(x.device)
#             cell = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels).to(x.device)
#             self.hidden_state = [(hidden, cell)]
#             self.last_x = torch.zeros(x.shape[0], 1, self.in_channels).to(x.device)
        
#         # [N T C H W]
#         # memory encoder
#         layer_output_list, last_state_list = self.encoder(x)

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
        self.encoder = nn.LSTM(in_channels, hidden_channels, num_layers, bidirectional=False, batch_first=True)
        self.decoder = nn.LSTM(in_channels, hidden_channels, num_layers, bidirectional=False, batch_first=True)
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
        if self.hidden is None:
        # if self.hidden is None or self.cell is None:
            self.hidden = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels).to(x.device)
            self.cell = torch.zeros(self.num_layers * self.direction, x.shape[0], self.hidden_channels).to(x.device)
            self.last_x = torch.zeros(x.shape[0], 1, self.in_channels).to(x.device)
        
        # [N T D]
        x = x.transpose(1, 2)
        # memory encoder
        seg_feature, (en_hidden, en_cell) = self.encoder(x, (self.hidden, self.cell))

        # if self.en_dropout is not None:
        #     frames_feature = self.en_dropout(frames_feature)  # [N * num_seg, in_channels]
        # frames_feature = x + frames_feature

        # memory decoder predict
        hidden = self.hidden
        cell = self.cell
        seg_feature_list = []
        for frame in range(x.shape[1]):
            if frame <= 0:
                input_frame = self.last_x
            else:
                input_frame = frame_feature
            frame_feature, (hidden, cell) = self.decoder(input_frame, (hidden, cell))
            seg_feature_list.append(frame_feature)
        seg_feature = torch.cat(seg_feature_list, dim=1)
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
    