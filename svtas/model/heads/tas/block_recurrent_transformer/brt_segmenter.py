'''
Author       : Thyssen Wen
Date         : 2023-02-28 08:47:36
LastEditors  : Thyssen Wen
LastEditTime : 2023-03-14 16:58:15
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/block_recurrent_transformer/brt_segmenter.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from ....builder import HEADS
from ...utils.attention_helper.attention_layer import MultiHeadAttention
from .block_recurrent_transformer import RecurrentStateGate
from .helper_function import (exists, FeedForward)
from ..asformer import ConvFeedForward

class RecurrentAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_state: int,
        kernel_size: int,
        dilation: int =1,
        state_len: int = 512,
        heads: int = 8,
        causal: bool = False,
    ):
        super().__init__()
        self.scale = dim ** -0.5

        self.dim = dim
        self.dim_state = dim_state

        self.heads = heads
        self.causal = causal
        self.state_len = state_len
        
        self.input_self_attn = MultiHeadAttention(embed_dim=dim, num_heads=heads, additional_mask_cfg={'type':'dilated_windows', 'window_size': kernel_size, 'dilation': dilation})
        self.state_self_attn = MultiHeadAttention(embed_dim=dim, num_heads=heads)

        self.input_state_cross_attn = MultiHeadAttention(embed_dim=dim, num_heads=heads, additional_mask_cfg={'type':'dilated_windows', 'window_size': state_len, 'dilation': dilation})
        self.state_input_cross_attn = MultiHeadAttention(embed_dim=dim, num_heads=heads)

        self.proj_gate = RecurrentStateGate(dim)
        self.ff_gate = RecurrentStateGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias = False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias = False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)

    def gen_key_padding_mask(self, masks):
        sum_masks = torch.sum(masks.squeeze(1), dim=-1) == 0.0
        key_padding_mask = masks.detach().clone()
        for bs in range(sum_masks.shape[0]):
            if sum_masks[bs]:
                key_padding_mask[bs] = 1.0
        return (key_padding_mask == 0.0).squeeze(1)
    
    def forward(
        self,
        x,
        masks,
        state=None,
    ):
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = torch.zeros((batch, self.state_len, self.dim_state), device=device)
        key_padding_mask = self.gen_key_padding_mask(masks=masks)

        input_attn, _ = self.input_self_attn(x, x, x, key_padding_mask)
        state_attn, _ = self.state_self_attn(state, state, state)

        input_as_q_cross_attn, _ = self.input_state_cross_attn(x, state, state, key_padding_mask)
        state_as_q_cross_attn, _ = self.state_input_cross_attn(state, x, x)

        projected_input = self.input_proj(torch.concat((input_as_q_cross_attn, input_attn), dim=2))
        projected_state = self.state_proj(torch.concat((state_as_q_cross_attn, state_attn), dim=2))

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state
    
class RecurrentHierarchicalAttentionLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 dim_state: int,
                 state_len: int = 512,
                 num_head: int = 8,
                 dilation: int = 8,
                 causal: bool = False,
                 dropout=0.0):
        super().__init__()
        self.att = RecurrentAttentionBlock(dim, dim_state, kernel_size=dilation, dilation=1, state_len=state_len, heads=num_head, causal=causal)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.chunck_size = dilation
        self.state = None
        self.state_len = state_len      
    
    def _clear_memory_buffer(self):
        self.state = None
    
    def forward(self, x, masks):
        x = torch.transpose(x, 1, 2)
        x, state = self.att(x, masks=masks, state=self.state)
        self.state = state.detach()

        # segmenter transformer
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x.transpose(1, 2) * masks[:, 0:1, :]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, state_len=512, num_head=1, stage='encoder', causal=False, dropout=0.0):
        super(AttModule, self).__init__()
        self.stage = stage
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        if self.stage == 'encoder':
            self.att_layer = RecurrentHierarchicalAttentionLayer(dim = out_channels,
                                                                dim_state = out_channels,
                                                                state_len = state_len,
                                                                num_head = num_head,
                                                                dilation= dilation,
                                                                causal = causal,
                                                                dropout=dropout)
        self.conv_1x1 = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout()
    
    def _clear_memory_buffer(self):
        if self.stage == 'encoder':
            self.att_layer._clear_memory_buffer()

    def forward(self, x, mask):
        out = self.feed_forward(x)
        if self.stage == 'encoder':
            out = self.att_layer(self.instance_norm(out), mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class Encoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, channel_masking_rate,
                 num_head=1, state_len=512, causal=False, dropout=0.0):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, state_len=state_len,
                       num_head=num_head, stage='encoder', causal=causal, dropout=dropout)
                       for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate
    
    def _clear_memory_buffer(self):
        for layer in self.layers:
            layer._clear_memory_buffer()

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes,
                 num_head=1, state_len=512, causal=False, dropout=0.0):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, state_len=state_len,
                       num_head=num_head, stage='decoder', causal=causal, dropout=dropout)
                       for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def _clear_memory_buffer(self):
        for layer in self.layers:
            layer._clear_memory_buffer()

    def forward(self, x, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

@HEADS.register()
class BRTSegmentationHead(nn.Module):
    """
    LinFormer Head for action segmentation
    """
    def __init__(self,
                 num_head=1,
                 state_len=512,
                 causal=False,
                 num_decoders=3,
                 encoder_num_layers=10,
                 decoder_num_layers=10,
                 num_f_maps=64,
                 input_dim=2048,
                 num_classes=11,
                 dropout=0.5,
                 channel_masking_rate=0.5,
                 sample_rate=1,
                 out_feature=False):
        super(BRTSegmentationHead, self).__init__()

        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = Encoder(encoder_num_layers, num_f_maps, input_dim, num_classes, channel_masking_rate,
                               num_head=num_head, state_len=state_len, causal=causal, dropout=dropout)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(decoder_num_layers, num_f_maps, num_classes, num_classes,
                                                             num_head=num_head, state_len=state_len, causal=causal, dropout=dropout))
                                                             for s in range(num_decoders)]) # num_decoders
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.encoder._clear_memory_buffer()
        for decoder in self.decoders:
            decoder._clear_memory_buffer()
    
    def forward(self, x, mask):
        # x.shape [N C T]
        # mask.shape [N C T]
        
        out, feature = self.encoder(x, mask[:, 0:1, ::self.sample_rate])
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, ::self.sample_rate], mask[:, 0:1, ::self.sample_rate])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        if self.out_feature is True:
            return feature, outputs
        return outputs
    