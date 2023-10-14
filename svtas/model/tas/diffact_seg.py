'''
Author       : Thyssen Wen
Date         : 2023-10-13 21:02:52
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 21:21:47
Description  : file content
FilePath     : /SVTAS/svtas/model/tas/diffact_seg.py
'''
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class DiffsusionActionSegmentationEncoderModel(nn.Module):
    """
    Diffusion Action Segmentation ref:https://arxiv.org/pdf/2303.17959.pdf
    """
    def __init__(self,
                 input_dim,
                 num_classes,
                 sample_rate,
                 num_layers = 10,
                 num_f_maps = 64,
                 kernel_size = 5, 
                 attn_dropout_rate = 0.5,
                 channel_dropout_rate = 0.5,
                 temporal_dropout_rate = 0.5,
                 feature_layer_indices = [5, 7, 9]):
        super(DiffsusionActionSegmentationEncoderModel, self).__init__()
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.feature_layer_indices = feature_layer_indices
        self.output_feature = False
        if feature_layer_indices is not None and len(feature_layer_indices) > 0:
            self.output_feature = True
        
        self.dropout_channel = nn.Dropout2d(p=channel_dropout_rate)
        self.dropout_temporal = nn.Dropout2d(p=temporal_dropout_rate)
        
        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)
        self.encoder = MixedConvAttModule(num_layers, num_f_maps, kernel_size, attn_dropout_rate)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def init_weights(self, init_cfg: dict = {}):
        pass
    
    def _clear_memory_buffer(self):
       pass
            
    def forward(self, x, mask):
        if self.output_feature:
            features = []
            if -1 in self.feature_layer_indices:
                features.append(x)
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            x, feature = self.encoder(self.conv_in(x), feature_layer_indices=self.feature_layer_indices)
            if feature is not None:
                features.append(feature)
            out = self.conv_out(x)
            if -2 in self.feature_layer_indices:
                features.append(F.softmax(out, 1))

            outputs = F.interpolate(
                input=out.unsqueeze(0),
                scale_factor=[1, self.sample_rate],
                mode="nearest")
            return dict(output=outputs, output_feature=torch.cat(features, 1))
        else:
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            out = self.conv_out(self.encoder(self.conv_in(x), feature_layer_indices=None))
            outputs = F.interpolate(
                input=out.unsqueeze(0),
                scale_factor=[1, self.sample_rate],
                mode="nearest")
            return dict(output=outputs)

class MixedConvAttModule(nn.Module): # for encoder
    def __init__(self, num_layers, num_f_maps, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModule, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.swish = nn.SiLU()
        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayer(num_f_maps, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i
    
    def forward(self, x, time_emb=None, feature_layer_indices=None):

        if time_emb is not None:
            x = x + self.time_proj(self.swish(time_emb))[:,:,None]

        if feature_layer_indices is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for l_id, layer in enumerate(self.layers):
                x = layer(x)
                if l_id in feature_layer_indices:
                    out.append(x)
            
            if len(out) > 0:
                out = torch.cat(out, 1)
            else:
                out = None

            return x, out
    

class MixedConvAttentionLayer(nn.Module):
    
    def __init__(self, d_model, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)
        
        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(x)
        x_k = self.att_linear_k(x)
        x_v = self.att_linear_v(x)
                
        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2) 
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x):
        
        x_drop = self.dropout(x)
        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop)
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out
