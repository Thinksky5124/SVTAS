'''
Author       : Thyssen Wen
Date         : 2023-10-24 19:04:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-24 20:14:41
Description  : file content
FilePath     : /SVTAS/svtas/model/unet/tas_diffusion_unet.py
'''
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .diffact_unet import get_timestep_embedding
from .condition_unet_1d import ConditionUnet1D
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class TASDiffusionConditionUnet(ConditionUnet1D):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 condition_dim,
                 time_embedding_dim,
                 condtion_res_layer_idx = [3, 4, 5, 6],
                 sample_rate = 1) -> None:
        super().__init__()
        self.single_condition_stage = SingleStageConditionModel(num_layers,
                                                                num_f_maps,
                                                                dim,
                                                                num_classes,
                                                                condition_dim,
                                                                time_embedding_dim,
                                                                condtion_res_layer_idx)
        self.sample_rate = sample_rate
        self.time_embedding_dim = time_embedding_dim

        self.time_in = nn.Sequential(
            torch.nn.Linear(time_embedding_dim, time_embedding_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embedding_dim, time_embedding_dim)
        )

    
    def forward(self, data_dict):
        # latent
        # timestep
        timestep = data_dict['timestep']
        noise_label = data_dict['noise_label']
        condition_latens = data_dict['condition_latens']
        mask = data_dict['masks_m']

        mask = mask[:, :, ::self.sample_rate]
        time_emb = get_timestep_embedding(timestep, self.time_embedding_dim)
        time_emb = self.time_in(time_emb).unsqueeze(0).permute(0, 2, 1)
        
        output = self.single_condition_stage(noise_label, condition_latens, time_emb, mask)
        outputs = output.unsqueeze(0)

        if self.training:
            outputs = F.interpolate(
                input=outputs,
                scale_factor=[1, self.sample_rate],
                mode="nearest")
        
        return dict(output=outputs)

class SingleStageConditionModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 condition_dim,
                 time_embedding_dim,
                 condtion_res_layer_idx = [3, 4, 5, 6],
                 out_feature=False):
        super(SingleStageConditionModel, self).__init__()
        self.out_feature = out_feature
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) if i not in condtion_res_layer_idx else
                                     copy.deepcopy(ConditionDilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, time_embedding_dim, condition_dim))
                                     for i in range(num_layers)])
        self.condtion_res_layer_idx = condtion_res_layer_idx
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, condition_latents, time_embedding, mask):
        feature_embedding = self.conv_1x1(x)
        feature = feature_embedding
        for i, layer in enumerate(self.layers):
            if i in self.condtion_res_layer_idx:
                feature = layer(feature, condition_latents, time_embedding, mask)
            else:
                feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        if self.out_feature is True:
            return feature_embedding * mask[:, 0:1, :], out

        return out

class ConditionDilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 time_embedding_dim,
                 condition_dim):
        super(ConditionDilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.condition_embedding = nn.Conv1d(condition_dim, out_channels, 1)
        self.time_embedding = nn.Conv1d(time_embedding_dim, out_channels, 1)
        self.swish = nn.SiLU()
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = nn.Dropout()

    def forward(self, x, condition_latents, time_embedding, mask):
        x = x + self.condition_embedding(condition_latents) + self.time_embedding(self.swish(time_embedding))
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]