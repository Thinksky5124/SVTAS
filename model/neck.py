'''
Author: Thyssen Wen
Date: 2022-03-25 10:29:18
LastEditors: Thyssen Wen
LastEditTime: 2022-03-28 21:02:05
Description: model neck
FilePath: /ETESVS/model/neck.py
'''
import torch
import torch.nn as nn
import math
from .head import SingleStageModel

class ETESVSNeck(nn.Module):
    def __init__(self,
                 pos_channels=2048,
                 num_layers=5,
                 num_f_maps=2048,
                 input_dim=2048,
                 output_dim=2048,
                 hidden_dim=4096,
                 dropout=0.5,
                 sample_rate=4,
                 clip_seg_num=15,
                 clip_buffer_num=0,
                 sliding_window=30):
        super().__init__()
        self.pos_channels = pos_channels
        self.num_layers = num_layers
        self.clip_seg_num = clip_seg_num
        self.clip_buffer_num = clip_buffer_num
        self.sliding_window = sliding_window
        self.sample_rate = sample_rate

        self.clip_windows = int(self.sliding_window // self.sample_rate)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.pe = PositionalEncoding(pos_channels)
        # self.ff = ForwardFedBlock(input_dim=input_dim,
        #             hidden_dim=hidden_dim,
        #             output_dim=output_dim,
        #             dropout=dropout)

    def init_weights(self):
        pass

    def forward(self, x, seg_mask, idx):
        # x.shape = [N * num_segs, 2048, 7, 7]
        x = self.avgpool(x)
        # x.shape = [N * num_segs, 2048, 1, 1]
        cls_feature = x

        # segmentation branch
        # [N * num_segs, 2048]
        seg_x = torch.squeeze(x)
        # [N, num_segs, 2048]
        seg_feature = torch.reshape(seg_x, shape=[-1, self.clip_seg_num, seg_x.shape[-1]])

        # position encoding
        # [N, num_segs, 2048]
        # p_idx = range(idx * self.clip_windows , idx * self.clip_windows + self.clip_seg_num)
        # seg_feature  = self.pe(seg_feature, p_idx)

        # seg_feature = self.ff(seg_feature, seg_mask)

        # memery
        # seg_feature = self.memery(seg_feature, seg_mask)

        # [N, 2048, num_segs]
        seg_feature = torch.permute(seg_feature, dims=[0, 2, 1])

        return seg_feature, cls_feature

class ForwardFedBlock(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 hidden_dim=2048,
                 output_dim=2048,
                 dropout=0.5):
        super().__init__()
        self.forward_fed_1 = nn.Linear(input_dim, hidden_dim)
        self.forward_fed_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x, mask):
        x = x * torch.permute(mask, dims=[0, 2, 1])
        return self.forward_fed_2(self.dropout(self.relu(self.forward_fed_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model, _weight=pe)

    def forward(self, x, p_idx):
        p_idx = torch.LongTensor(list(p_idx)).to(x.device)
        pe = self.pe(p_idx).unsqueeze(0).repeat(x.shape[0], 1, 1)
        # x = torch.concat([x, pe], dim=2)
        x = x + pe
        return self.dropout(x)
        
class MemeryLayer(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 input_dim,
                 output_dim,
                 clip_seg_num=15,
                 sliding_window=30,
                 sample_rate=4,
                 clip_buffer_num=0):
        super().__init__()
        # self.w_f = OverLapCausalConvBlock(num_layers,
        #          num_f_maps,
        #          input_dim,
        #          output_dim,
        #          clip_seg_num=clip_seg_num,
        #          sliding_window=sliding_window,
        #          sample_rate=sample_rate,
        #          activation='sigmoid')
        
        # self.w_i = OverLapCausalConvBlock(num_layers,
        #          num_f_maps,
        #          input_dim,
        #          output_dim,
        #          clip_seg_num=clip_seg_num,
        #          sliding_window=sliding_window,
        #          sample_rate=sample_rate,
        #          activation='sigmoid')
        
        # self.w = OverLapCausalConvBlock(num_layers,
        #          num_f_maps,
        #          input_dim,
        #          output_dim,
        #          clip_seg_num=clip_seg_num,
        #          sliding_window=sliding_window,
        #          sample_rate=sample_rate,
        #          activation='tanh')
        
        # self.w_o = OverLapCausalConvBlock(num_layers,
        #          num_f_maps,
        #          input_dim,
        #          output_dim,
        #          clip_seg_num=clip_seg_num,
        #          sliding_window=sliding_window,
        #          sample_rate=sample_rate,
        #          activation='sigmoid')
        # self.tanh = nn.Tanh()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.c_t = None
        self.h_t = None
        self.buffer_mask = None
    
    def _clear_memery_buffer(self):
        self.c_t = None
        self.h_t = None
        self.buffer_mask = None

    def forward(self, x, mask):
        if self.c_t is None or self.h_t is None:
            self.c_t = torch.ones((self.num_layers, x.shape[0], self.input_dim), device=x.device)
            self.h_t = torch.ones((self.num_layers, x.shape[0], self.input_dim), device=x.device)
            self.buffer_mask = torch.zeros(mask.shape, device=mask.device)
        
        # # embedding
        # z_f = self.w_f(x, self.h_t, mask, self.buffer_mask)
        # z_i = self.w_i(x, self.h_t, mask, self.buffer_mask)
        # z = self.w(x, self.h_t, mask, self.buffer_mask)
        # z_o = self.w_o(x, self.h_t, mask, self.buffer_mask)

        # # memery
        # c_t = z_f * self.c_t + z_i * z
        # h_t = self.tanh(c_t) * z_o

        # # memery
        # self.h_t = h_t.detach().clone()
        # self.c_t = c_t.detach().clone()
        # self.buffer_mask = mask.detach().clone()

        # fusion_feature = torch.concat([
        #     self.h_t[:, :, :self.mbf_start],
        #     (self.h_t[:, :, self.mbf_start:] + x[:, :, :self.over_lap_len]) / 2.0,
        #     x[:, :, self.over_lap_len:]],
        #     dim=2)
        # fusion_mask = torch.concat([
        #     self.buffer_mask[:, :, :self.mbf_start],
        #     (self.buffer_mask[:, :, self.mbf_start:] + mask[:, :, :self.over_lap_len]) / 2.0,
        #     mask[:, :, self.over_lap_len:]],
        #     dim=2)
        mask = torch.permute(mask, dims=[0, 2, 1])
        output, (h_t, c_t) = self.lstm(x * mask, (self.h_t, self.c_t))
        self.h_t = h_t.detach().clone()
        self.c_t = c_t.detach().clone()
        return output

class OverLapCausalConvBlock(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 input_dim,
                 output_dim,
                 clip_seg_num=15,
                 sliding_window=30,
                 sample_rate=4,
                 activation='sigmoid'):
        super().__init__()
        if activation not in ['sigmoid', 'tanh']:
            raise NotImplementedError

        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.sample_rate = sample_rate
        self.over_lap_len = int(self.clip_seg_num - self.sliding_window // self.sample_rate)
        if self.over_lap_len < 0:
            self.over_lap_len = 0
        self.mbf_start = self.clip_seg_num - self.over_lap_len

        self.causal_conv = SingleStageModel(num_layers, num_f_maps, input_dim, output_dim)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
    
    def forward(self, x, memery_buffer, mask, mask_buffer):
        # x shape [N C T]
        # memery_buffer shape [N C T]
        # mask shape [N 1 T]
        # mask_buffer shape [N 1 T]

        # step 1 feature refine
        fusion_feature = torch.concat([
            memery_buffer[:, :, :self.mbf_start],
            (memery_buffer[:, :, self.mbf_start:] + x[:, :, :self.over_lap_len]) / 2.0,
            x[:, :, self.over_lap_len:]],
            dim=2)
        fusion_mask = torch.concat([
            mask_buffer[:, :, :self.mbf_start],
            (mask_buffer[:, :, self.mbf_start:] + mask[:, :, :self.over_lap_len]) / 2.0,
            mask[:, :, self.over_lap_len:]],
            dim=2)

        # step 2 feature conv
        x = self.causal_conv(fusion_feature, fusion_mask)

        # step 3 activation
        x = self.activation(x)

        # feature clap
        # x shape [N C T]
        x = x[:, :, -self.clip_seg_num:]
        return x