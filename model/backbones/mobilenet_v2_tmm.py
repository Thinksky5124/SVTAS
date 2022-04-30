'''
Author: Thyssen Wen
Date: 2022-04-30 14:02:02
LastEditors: Thyssen Wen
LastEditTime: 2022-04-30 14:10:34
Description: MobileNet V2 temporal memory module
FilePath: /ETESVS/model/backbones/mobilenet_v2_tmm.py
'''
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

class Conv3DLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias):
        super(Conv3DLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        
        # deepwise conv
        conv_cfg=dict(type='Conv3d')
        norm_cfg=dict(type='BN3d')
        act_cfg=dict(type='ReLU6')
        self.conv = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels + self.hidden_channels,
                out_channels=self.in_channels + self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                groups=self.in_channels + self.hidden_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=self.in_channels + self.hidden_channels,
                out_channels=4 * self.hidden_dim,
                kernel_size=1,
                padding=0,
                bias=self.bias,
                groups=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None))
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))