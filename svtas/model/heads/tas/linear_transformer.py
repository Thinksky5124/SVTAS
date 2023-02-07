'''
Author       : Thyssen Wen
Date         : 2022-10-22 13:56:11
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-07 13:42:26
Description  : Linear Head
FilePath     : /SVTAS/svtas/model/heads/tas/linear_transformer.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from .asformer import ConvFeedForward, exponential_descrease

from ...builder import HEADS

def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
        return conv
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1/dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin
    
class LinAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, bl, stage, att_type, r1, r2, r3, method="learnable"): # r1 = r2
        super(LinAttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.dropout = nn.Dropout()
        self.P_bar = None
        self.E = get_EF(k_dim // r1, k_dim // r1, method, q_dim // r1)
        self.F = get_EF(v_dim // r3, v_dim // r3, method, q_dim // r1)
        self.causal_mask = None
        # self.causal_mask = self.gen_causal_mask(q_dim // r1, k_dim // r2)
        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        self.is_proj_tensor = isinstance(self.E, torch.Tensor)
        assert self.att_type in ['linear_att']
        assert self.stage in ['encoder','decoder']
    
    def gen_causal_mask(self, input_size, dim_k, full_attention=False):
        """
        Generates a causal mask of size (input_size, dim_k) for linformer
        Else, it generates (input_size, input_size) for full attention
        """
        if full_attention:
            return (torch.triu(torch.ones(input_size, input_size))==1).transpose(0,1)
        return (torch.triu(torch.ones(dim_k, input_size))==1).transpose(0,1)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'linear_att':
            return self._linear_self_att(query, key, value, mask)
    
    def _linear_self_att(self, Q, K, V, mask):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        ref: https://github.com/tatp22/linformer-pytorch/blob/master/linformer_pytorch/linformer_pytorch.py
        """

        K = K.transpose(1,2)

        if self.is_proj_tensor:
            self.E = self.E.to(K.device)
            K = torch.matmul(K, self.E)
        else:
            K = self.E(K)


        Q = torch.matmul(Q, K)

        P_bar = Q/torch.sqrt(torch.tensor(Q.shape[-1]).type(Q.type())).to(Q.device)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, float('-inf'))
        P_bar = P_bar.softmax(dim=-1)

        P_bar = self.dropout(P_bar)

        V = V.transpose(1,2)
        if self.is_proj_tensor:
            self.F = self.F.to(V.device)
            V = torch.matmul(V, self.F)
        else:
            V = self.F(V)

        V = V.transpose(1,2)
        out_tensor = torch.matmul(P_bar, V)

        out_tensor = self.conv_out(F.relu(out_tensor))
        return out_tensor * mask[:, 0:1, :]


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = LinAttLayer(q_dim=in_channels, k_dim=in_channels, v_dim=out_channels, bl=dilation, r1=r1, r2=r1, r3=r2, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
             range(num_layers)])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

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
        # feature = self.position_en(feature)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

@HEADS.register()
class LinformerHead(nn.Module):
    """
    LinFormer Head for action segmentation
    """
    def __init__(self,
                 num_decoders=3,
                 num_layers=10,
                 num_f_maps=64,
                 input_dim=2048,
                 num_classes=11,
                 channel_masking_rate=0.5,
                 sample_rate=1,
                 r1=2,
                 r2=2,
                 out_feature=False):
        super(LinformerHead, self).__init__()

        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='linear_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='linear_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
    
    def forward(self, x, mask):
        # x.shape [N C T]
        # mask.shape [N C T]
        
        out, feature = self.encoder(x, mask[:, 0:1, ::self.sample_rate])
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, ::self.sample_rate], feature* mask[:, 0:1, ::self.sample_rate], mask[:, 0:1, ::self.sample_rate])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        if self.out_feature is True:
            return feature, outputs
        return outputs