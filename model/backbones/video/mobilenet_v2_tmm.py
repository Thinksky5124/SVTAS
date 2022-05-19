'''
Author: Thyssen Wen
Date: 2022-04-30 14:02:02
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 21:33:33
Description: MobileNet V2 temporal memory module
FilePath     : /ETESVS/model/backbones/video/mobilenet_v2_tmm.py
'''
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from utils.logger import get_logger
from mmcv.cnn import ConvModule
from .mobilenet_v2_tsm import MobileNetV2TSM
from ...builder import BACKBONES

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
                out_channels=4 * self.hidden_channels,
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
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, clip_size):
        temporal, height, width = clip_size
        return (torch.zeros(batch_size, self.hidden_channels, temporal, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, temporal, height, width, device=self.conv.weight.device))

class Conv3DLSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 num_layers,
                 batch_first=True,
                 bias=True):
        super(Conv3DLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_channels` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.in_channels if i == 0 else self.hidden_channels[i - 1]

            cell_list.append(Conv3DLSTMCell(in_channels=cur_input_dim,
                                          hidden_channels=self.hidden_channels[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, t, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             clip_size=(t, h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input,
                                                cur_state=[h, c])

            layer_output = h
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list[-1], last_state_list[-1]

    def _init_hidden(self, batch_size, clip_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, clip_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class TemporalMemoryBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 num_layers,
                 num_segments,
                 batch_first=True,
                 bias=True):
        super().__init__()
        self.num_segments = num_segments
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.rnn = Conv3DLSTM(in_channels=in_channels,
                 hidden_channels=hidden_channels,
                 kernel_size=kernel_size,
                 num_layers=num_layers,
                 batch_first=batch_first,
                 bias=bias)
        
        self.memory_states = None
    
    def _init_hidden(self, batch_size, clip_size, device):
        self.memory_states = []
        for i in range(self.num_layers):
            temporal, height, width = clip_size
            self.memory_states = [[torch.zeros(batch_size, self.hidden_channels, temporal, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, temporal, height, width, device=device)] for _ in range(self.num_layers)]
        return self.memory_states
    
    def _resert_memory(self):
        self.memory_states = None
    
    def forward(self, x):
        # in x.shape [N*T, C, H, W]
        # x.shape [N, C, T, H, W]
        re_x = torch.reshape(x, shape=[-1, self.num_segments] + list(x.shape[1:])).transpose(1, 2)
        if self.memory_states is None:
            self.memory_states = self._init_hidden(re_x.shape[0], re_x.shape[2:], re_x.device)
        layer_output, last_state_list = self.rnn(re_x, self.memory_states)
        # layer_output.shape [N, C, T, H, W]
        layer_output = layer_output.transpose(1, 2)
        # layer_output.shape [N, T, C, H, W]
        x = torch.reshape(layer_output, shape=x.shape)

        # memory
        self.hidden_state = [[last_state_list[l][i].detach().clone() for i in range(len(last_state_list[l]))] for l in range(self.num_layers)]
        return x

@BACKBONES.register()
class MobileNetV2TMM(MobileNetV2TSM):
    def __init__(self,
                 is_memory=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.is_memory = is_memory
    
    def make_temporal_memory(self):
        def make_layer(stage):
            blocks = list(stage.children())
            conv3dlstm = TemporalMemoryBlock(
                            in_channels=blocks[-1].conv[-1].out_channels,
                            hidden_channels=blocks[-1].conv[-1].out_channels,
                            kernel_size=(3, 3, 3),
                            num_layers=1,
                            num_segments=self.num_segments)
            add_blocks = blocks + [conv3dlstm]
                
            return  nn.Sequential(*add_blocks)

        self.layer3 = make_layer(self.layer3)
        self.layer4 = make_layer(self.layer4)
        self.layer5 = make_layer(self.layer5)
        self.layer6 = make_layer(self.layer6)
    
    def _clear_memory_buffer(self):
        self.apply(self._clean_buffers)

    @staticmethod
    def _clean_buffers(m):
        if issubclass(type(m), TemporalMemoryBlock):
            m._resert_memory()

    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if self.is_memory:
            self.make_temporal_memory()
        
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)
            elif self.pretrained is None:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                        constant_init(m, 1)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                        constant_init(m, 1)