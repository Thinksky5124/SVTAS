'''
Author       : Thyssen Wen
Date         : 2022-05-16 11:08:52
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-16 20:51:07
Description  : SpatioTemporalLSTMCell Layer ref:https://github.com/thuml/predrnn-pytorch/blob/master/core/layers/SpatioTemporalLSTMCell_v2.py
FilePath     : /ETESVS/model/backbones/utils/stlstm/stlstm.py
'''
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm, is_deep_wise_conv=True):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if is_deep_wise_conv is False:
            if layer_norm:
                self.conv_x = nn.Sequential(
                    nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                    nn.LayerNorm([num_hidden * 7, width, width])
                )
                self.conv_h = nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                    nn.LayerNorm([num_hidden * 4, width, width])
                )
                self.conv_m = nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                    nn.LayerNorm([num_hidden * 3, width, width])
                )
                self.conv_o = nn.Sequential(
                    nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                    nn.LayerNorm([num_hidden, width, width])
                )
            else:
                self.conv_x = nn.Sequential(
                    nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                )
                self.conv_h = nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                )
                self.conv_m = nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                )
                self.conv_o = nn.Sequential(
                    nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                )
        else:
            if layer_norm:
                conv_cfg=dict(type='Conv2d')
                norm_cfg=dict(type='LN')
                act_cfg=dict(type='ReLU6')
                self.conv_x = nn.Sequential(
                    ConvModule(in_channels=in_channel, out_channels=in_channel, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=in_channel, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=in_channel, out_channels=num_hidden * 7, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
                self.conv_h = nn.Sequential(
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=num_hidden, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden * 4, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
                self.conv_m = nn.Sequential(
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=num_hidden, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden * 3, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
                self.conv_o = nn.Sequential(
                    ConvModule(in_channels=num_hidden * 2, out_channels=num_hidden * 2, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=num_hidden * 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=num_hidden * 2, out_channels=num_hidden, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
            else:
                conv_cfg=dict(type='Conv2d')
                norm_cfg=None
                act_cfg=dict(type='ReLU6')
                self.conv_x = nn.Sequential(
                    ConvModule(in_channels=in_channel, out_channels=in_channel, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=in_channel, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=in_channel, out_channels=num_hidden * 7, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
                self.conv_h = nn.Sequential(
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=num_hidden, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden * 4, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
                self.conv_m = nn.Sequential(
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=num_hidden, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=num_hidden, out_channels=num_hidden * 3, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
                self.conv_o = nn.Sequential(
                    ConvModule(in_channels=num_hidden * 2, out_channels=num_hidden * 2, kernel_size=filter_size, padding=self.padding, stride=stride, bias=False, groups=num_hidden * 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(in_channels=num_hidden * 2, out_channels=num_hidden, kernel_size=1, padding=0, bias=False, groups=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m