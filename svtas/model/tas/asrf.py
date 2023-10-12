'''
Author       : Thyssen Wen
Date         : 2023-02-08 10:38:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-08 20:56:27
Description  : ASRF ref:https://github.com/yiskw713/asrf/blob/main/libs/models/tcn.py
FilePath     : /SVTAS/svtas/model/heads/tas/asrf.py
'''
# Originally written by yabufarha
# https://github.com/yabufarha/ms-tcn/blob/master/model.py

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from svtas.utils import AbstractBuildFactory
from .mstcn import DilatedResidualLayer, SingleStageModel

@AbstractBuildFactory.register('model')
class ActionSegmentRefinementFramework(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 2048
        n_feature: 64
        num_classes: the number of action classes
        num_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        num_features: int,
        num_classes: int,
        num_stages: int,
        num_layers: int,
        num_stages_asb: Optional[int] = None,
        num_stages_brb: Optional[int] = None,
        sample_rate: int = 1,
        out_feature: bool = False
    ) -> None:
        self.sample_rate = sample_rate
        if not isinstance(num_stages_asb, int):
            num_stages_asb = num_stages

        if not isinstance(num_stages_brb, int):
            num_stages_brb = num_stages

        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, num_features, 1)
        shared_layers = [
            DilatedResidualLayer(2 ** i, num_features, num_features)
            for i in range(num_layers)
        ]
        self.shared_layers = nn.ModuleList(shared_layers)
        self.conv_cls = nn.Conv1d(num_features, num_classes, 1)
        self.conv_bound = nn.Conv1d(num_features, 1, 1)

        # action segmentation branch
        asb = [
            SingleStageModel(num_layers, num_features, num_classes, num_classes)
            for _ in range(num_stages_asb - 1)
        ]

        # boundary regression branch
        brb = [
            SingleStageModel(num_layers, num_features, 1, 1) for _ in range(num_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask[:, :, ::self.sample_rate]
        
        out = self.conv_in(x)
        for layer in self.shared_layers:
            out = layer(out, mask)

        out_cls = self.conv_cls(out)
        out_bound = self.conv_bound(out)

        outputs_cls = out_cls.unsqueeze(0)
        outputs_bound = out_bound.unsqueeze(0)

        for as_stage in self.asb:
            out_cls = as_stage(self.activation_asb(out_cls) * mask[:, 0:1, :], mask)
            outputs_cls = torch.cat((outputs_cls, out_cls.unsqueeze(0)), dim=0)

        for br_stage in self.brb:
            out_bound = br_stage(self.activation_brb(out_bound) * mask[:, 0:1, :], mask)
            outputs_bound = torch.cat((outputs_bound, out_bound.unsqueeze(0)), dim=0)
        
        outputs_cls = F.interpolate(
            input=outputs_cls,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        outputs_bound = F.interpolate(
            input=outputs_bound,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        return {"cls":outputs_cls, "boundary":outputs_bound, "features":x.detach().clone()}