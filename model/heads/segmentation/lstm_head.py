'''
Author       : Thyssen Wen
Date         : 2022-05-13 15:25:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-13 20:38:03
Description  : BiLSTM for action segmentation reproduce for:https://openaccess.thecvf.com/content_cvpr_2016/papers/Singh_A_Multi-Stream_Bi-Directional_CVPR_2016_paper.pdf
FilePath     : /ETESVS/model/heads/lstm_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import HEADS

@HEADS.register()
class LSTMSegmentationHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 sample_rate=1,
                 hidden_channels=1024,
                 num_layers=3,
                 batch_first=True,
                 dropout=0.5,
                 bidirectional=True,
                 is_memory_sliding=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_memory_sliding = is_memory_sliding
        self.sample_rate = sample_rate

        self.num_directions = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_channels,
                            num_layers=num_layers,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        
        self.fc_cls = nn.Linear(self.hidden_channels, self.num_classes)

        self.memory_hidden_list = None
        
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.memory_hidden_list = None
    
    def forward(self, x, masks):
        # x [N D T]
        # mask [N D T]
        clip_seg_num = x.shape[-1]
        batch_size = x.shape[0]

        if self.memory_hidden_list is None or self.is_memory_sliding is False:
            self.memory_hidden_list = [
                torch.zeros([self.num_directions * self.num_layers, batch_size, self.hidden_channels]).to(x.device),
                torch.zeros([self.num_directions * self.num_layers, batch_size, self.hidden_channels]).to(x.device)
            ]
        
        x_transpose = x.transpose(1, 2)
        # output [N, clip_seg_num, hidden_head_num * hidden_channels]
        output, (hn, cn) = self.lstm(x_transpose, (self.memory_hidden_list[0], self.memory_hidden_list[1]))

        # memory
        self.memory_hidden_list[0] = hn.detach().clone()
        self.memory_hidden_list[1] = cn.detach().clone()

        # output [N, clip_seg_num, num_directions, hidden_channels] 0 forward 1 backward
        output = output.view(batch_size, clip_seg_num, self.num_directions, self.hidden_channels)
        # output [N, clip_seg_num, hidden_channels]
        output = torch.mean(output, dim=2)
        # output [N * clip_seg_num, hidden_channels]
        output = torch.reshape(output, [-1, self.hidden_channels])

        score = self.fc_cls(output)

        # [N, num_class, num_seg]
        score = torch.reshape(
            score, [-1, clip_seg_num, self.num_classes]).permute([0, 2, 1])
        score = score * masks[: ,0:1, ::self.sample_rate]

        # [stage_num, N, C, T]
        score = score.unsqueeze(0)
        score = F.interpolate(
            input=score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        return score