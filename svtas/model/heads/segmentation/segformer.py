'''
Author       : Thyssen Wen
Date         : 2022-12-22 20:15:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-28 20:03:22
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/segmentation/segformer.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from ...builder import HEADS

class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        # deal with d_model is odd number
        if d_model % 2 != 0:
            norm_d_model = d_model + 1
        else:
            norm_d_model = d_model
        # position encoding
        pe = torch.zeros(max_len, norm_d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, norm_d_model, 2).float() * -(math.log(10000.0) / norm_d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [d_model max_len]
        pe = pe.transpose(0, 1)[:d_model, :]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: [N C T]
        x = x + self.pe[:, :x.shape[-1]].unsqueeze(0)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 embed_dim,
                 dropout=0.0) -> None:
        super().__init__()
        if dropout > 0.0:
            self.mlp = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 1),
                nn.Conv1d(embed_dim, embed_dim, 1),
                nn.GELU(),
                nn.Dropout(p=dropout))
        else:
            self.mlp = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 1),
                nn.Conv1d(embed_dim, embed_dim, 1),
                nn.GELU())
        
    def forward(self, x):
        return self.mlp(x)

class ResdualDilationConvBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 dilation,
                 dropout=0.0) -> None:
        super().__init__()
        self.dialtion_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )
        if dropout > 0.0:
            self.norm = nn.Dropout(p=dropout)
        else:
            self.norm = None
        
    def forward(self, x):
        out = self.dialtion_conv(x) + x
        if self.norm is not None:
            return self.norm(out)
        return out
        
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 need_causal_mask=False) -> None:
        super().__init__()
        self.att_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.need_causal_mask = need_causal_mask
        self.num_heads = num_heads
    
    def gen_causal_mask(self, input_size, masks):
        """
        Generates a causal mask of size (input_size, input_size) for attention
        """
        # [T, T]
        l_l_mask = torch.triu(torch.ones(input_size, input_size), diagonal=1) == 1
        l_l_mask = l_l_mask.to(masks.device)

        # [N,1,T] -> [N,T,T]
        sum_masks = torch.sum(masks.squeeze(1), dim=-1) == 0.0
        for bs in range(sum_masks.shape[0]):
            if sum_masks[bs]:
                masks[bs] = 1.0
        masks_matrix = torch.bmm(masks.transpose(1, 2), masks)
        att_mask = masks_matrix * l_l_mask.unsqueeze(0)

        # [N,T,T] -> [N*num_heads,T,T]
        att_mask = torch.repeat_interleave(att_mask, repeats=self.num_heads, dim=0)
        return att_mask
    
    def gen_key_padding_mask(self, masks):
        sum_masks = torch.sum(masks.squeeze(1), dim=-1) == 0.0
        for bs in range(sum_masks.shape[0]):
            if sum_masks[bs]:
                masks[bs] = 1.0
        return (masks == 0.0).squeeze(1)
    
    def forward(self, q, k, v, masks):
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        key_padding_mask = self.gen_key_padding_mask(masks=masks)
        if self.need_causal_mask:
            causal_mask = self.gen_causal_mask(q.shape[1], masks)
        else:
            causal_mask = None
        
        out, att_map = self.att_layer(q, k, v, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
        out = torch.transpose(out, 1, 2)
        return out * masks

class ResdualMultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 need_causal_mask=False) -> None:
        super().__init__()
        self.att_layer = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
                                                        dropout=dropout, need_causal_mask=need_causal_mask)
        self.prev_norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.feed_forward = FeedForwardNetwork(embed_dim=embed_dim, dropout=dropout)
        self.after_norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.need_causal_mask = need_causal_mask
    
    def forward(self, x, q, k, v, masks):
        out = self.att_layer(q, k, v, masks)
        out = self.prev_norm(out + x)
        out = self.feed_forward(out) + out
        out = self.after_norm(out)
        return out
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dilation=1,
                 dropout=0.0) -> None:
        super().__init__()
        self.dilation_conv = ResdualDilationConvBlock(embed_dim=embed_dim, dilation=dilation, dropout=dropout)
        self.att_block = ResdualMultiHeadAttentionBlock(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 dropout=dropout)
    
    def forward(self, x, masks):
        x = self.dilation_conv(x)
        x = self.att_block(x, x, x, x, masks)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dilation=1,
                 dropout=0.0,
                 need_causal_mask=True) -> None:
        super().__init__()
        self.dilation_conv = ResdualDilationConvBlock(embed_dim=embed_dim, dilation=dilation, dropout=dropout)
        self.att_layer = MultiHeadAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
                                                        dropout=dropout, need_causal_mask=need_causal_mask)
        self.norm = nn.InstanceNorm1d(num_features=embed_dim)
        self.att_block = ResdualMultiHeadAttentionBlock(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 dropout=dropout,
                                                 need_causal_mask=need_causal_mask)
        self.need_causal_mask = need_causal_mask
    
    def forward(self, x, v_x, masks):
        x = self.dilation_conv(x)
        v_x_a = self.att_layer(v_x, v_x, v_x, masks)
        v_x = self.norm(v_x_a + v_x)
        x = self.att_block(v_x, x, x, v_x, masks)
        return x

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_heads,
                 num_layers,
                 num_classes,
                 dropout=0.0):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, embed_dim, 1)
        self.pe_layer = PositionalEncoding(d_model=embed_dim)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, dilation=2 ** i)
                for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(embed_dim, num_classes, 1)

    def forward(self, x, mask):
        feature = self.conv_1x1(x)
        # position encoding
        feature = self.pe_layer(feature)
        for layer in self.layers:
            feature = layer(feature, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_heads,
                 num_layers,
                 num_classes,
                 dropout=0.0):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, embed_dim, 1)
        self.pe_layer = PositionalEncoding(d_model=embed_dim)
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, dilation=2 ** i)
                for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(embed_dim, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        # position encoding
        feature = self.pe_layer(feature)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
        
@HEADS.register()
class SegFormer(nn.Module):
    def __init__(self,
                 in_channels,
                 num_decoders=3,
                 num_layers=10,
                 num_classes=11,
                 input_dropout=0.5,
                 embed_dim=64,
                 num_heads=8,
                 dropout=0.0,
                 sample_rate=1,
                 out_feature=False):
        super(SegFormer, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = Encoder(in_channels=in_channels,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               num_classes=num_classes,
                               dropout=dropout)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(in_channels=num_classes,
                                                             embed_dim=embed_dim,
                                                             num_heads=num_heads,
                                                             num_layers=num_layers,
                                                             num_classes=num_classes,
                                                             dropout=dropout))
                                                             for s in range(num_decoders)]) # num_decoders
        self.dropout = nn.Dropout(p=input_dropout)
        self.input_dropout = input_dropout
        
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
                
    def forward(self, x, mask):
        # x.shape [N C T]
        # mask.shape [N C T]
        if self.input_dropout > 0:
            x = self.dropout(x)
        
        out, feature = self.encoder(x, mask[:, 0:1, ::self.sample_rate])
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1), feature, mask[:, 0:1, ::self.sample_rate])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest") * mask[:, 0:1].unsqueeze(0)

        if self.out_feature is True:
            return feature, outputs
        return outputs