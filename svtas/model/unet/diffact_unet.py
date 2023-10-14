'''
Author       : Thyssen Wen
Date         : 2023-10-12 16:26:13
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 21:32:10
Description  : ref:https://github.com/Finspire13/DiffAct/blob/main/model.py
FilePath     : /SVTAS/svtas/model/unet/diffact_unet.py
'''
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .condition_unet_1d import ConditionUnet1D
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class DiffsusionActionSegmentationConditionUnet(ConditionUnet1D):
    """
    Diffusion Action Segmentation ref:https://arxiv.org/pdf/2303.17959.pdf
    """
    def __init__(self,
                 input_dim,
                 num_classes,
                 sample_rate = 1,
                 num_layers = 8,
                 num_f_maps = 24,
                 kernel_size = 5,
                 attn_dropout_rate = 0.1,
                 time_emb_dim = 512,
                 ignore_index = -100,
                 condition_types = ['full', 'zero', 'boundary03-', 'segment=1', 'segment=1']) -> None:
        super().__init__()
        self.decoder = DiffsusionActionSegmentationConditionUnetUpModel(input_dim = input_dim,
                                                                        num_classes = num_classes,
                                                                        num_layers = num_layers,
                                                                        num_f_maps = num_f_maps,
                                                                        time_emb_dim = time_emb_dim,
                                                                        kernel_size = kernel_size,
                                                                        attn_dropout_rate = attn_dropout_rate)
        self.condition_types = condition_types
        self.ignore_index = ignore_index
        self.sample_rate = sample_rate
    
    def get_random_label_index(self, labels):
        y = torch.zeros_like(labels)
        refine_labels = torch.where(labels != self.ignore_index, labels, y)
        events = torch.unique(refine_labels, dim=-1)
        random_index = torch.argsort(torch.rand_like(events, dtype=torch.float), dim=-1)
        random_event = torch.gather(events, dim=-1, index=random_index)[:, :1]
        return random_event
        
    def get_condition_latent_mask(self, condition_latens, boundary_prob, labels):
        cond_type = random.choice(self.condition_types)

        if cond_type == 'full':
            feature_mask = torch.ones_like(condition_latens)
        
        elif cond_type == 'zero':
            feature_mask = torch.zeros_like(condition_latens)
        
        elif cond_type == 'boundary05-':
            feature_mask = (boundary_prob < 0.5)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, self.sample_rate]
            

        elif cond_type == 'boundary03-':
            feature_mask = (boundary_prob < 0.3)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, self.sample_rate]

        elif cond_type == 'segment=1':
            random_event = self.get_random_label_index(labels=labels)
            feature_mask = (labels != random_event) * (labels != self.ignore_index)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, self.sample_rate]

        elif cond_type == 'segment=2':
            random_event_1 = self.get_random_label_index(labels=labels)
            random_event_2 = self.get_random_label_index(labels=labels)
            while random_event_1 == random_event_2:
                random_event_2 = self.get_random_label_index(labels=labels)
            feature_mask = (labels != random_event_1) * (labels != random_event_2) * (labels != self.ignore_index)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, self.sample_rate]
        else:
            raise Exception('Invalid Cond Type')
        return feature_mask
    
    def forward(self, data_dict):
        # latent
        # timestep
        timestep = data_dict['timestep']
        noise_label = data_dict['noise_label'][:, :, ::self.sample_rate]
        condition_latens = data_dict['condition_latens']
        # unet down sample stage

        if self.training:
            gt_labels = data_dict['labels']
            boundary_prob = data_dict['boundary_prob']
            feature_mask = self.get_condition_latent_mask(condition_latens=condition_latens,
                                                          labels=gt_labels,
                                                          boundary_prob=boundary_prob)

            data_dict = dict(
                condition_latens = condition_latens * feature_mask,
                noise_label = noise_label,
                timestep = timestep
            )
        else:
            data_dict = dict(
                condition_latens = condition_latens,
                noise_label = noise_label,
                timestep = timestep
            )
        pred_label_dict = self.decoder(data_dict)
        pred_label = pred_label_dict['output']
        pred_label = F.interpolate(
                input=pred_label.unsqueeze(0),
                scale_factor=[1, self.sample_rate],
                mode="nearest")
        return dict(output = pred_label)


def get_timestep_embedding(timesteps, embedding_dim): # for diffusion model
    # timesteps: batch,
    # out:       batch, embedding_dim
    """
    This matches the implementation in Denoising DiffusionModel Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb
        
class DiffsusionActionSegmentationConditionUnetUpModel(nn.Module):
    """
    Diffusion Action Segmentation ref:https://arxiv.org/pdf/2303.17959.pdf
    """
    def __init__(self,
                 input_dim,
                 num_classes,
                 num_layers = 8,
                 num_f_maps = 24,
                 time_emb_dim = 512,
                 kernel_size = 5,
                 attn_dropout_rate = 0.1):
        
        super(DiffsusionActionSegmentationConditionUnetUpModel, self).__init__()

        self.time_emb_dim = time_emb_dim

        self.time_in = nn.Sequential(
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)
        self.module = MixedConvAttModuleV2(num_layers, num_f_maps, input_dim, kernel_size, attn_dropout_rate, time_emb_dim)
        self.conv_out =  nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, data_dict):
        condition_latens, timestep, noise_label = data_dict['condition_latens'], data_dict['timestep'], data_dict['noise_label']

        time_emb = get_timestep_embedding(timestep, self.time_emb_dim)
        time_emb = self.time_in(time_emb)

        fra = self.conv_in(noise_label)
        fra = self.module(fra, condition_latens, time_emb)
        pred = self.conv_out(fra)
        return dict(output=pred)


class MixedConvAttModuleV2(nn.Module): # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)
        self.swish = nn.SiLU()
        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i
    
    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(self.swish(time_emb))[:,:,None]

        for layer in self.layers:
            x = layer(x, x_cross)

        return x


class MixedConvAttentionLayerV2(nn.Module):
    
    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()
        
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
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
        
        
    def attention(self, x, x_cross):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
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
    
                
    def forward(self, x, x_cross):
        
        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)
                
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out

