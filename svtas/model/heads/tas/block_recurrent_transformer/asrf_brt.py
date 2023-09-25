'''
Author       : Thyssen Wen
Date         : 2023-02-26 20:14:37
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-26 20:39:10
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/block_recurrent_transformer/asrf_brt.py
'''
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ....builder import HEADS
from .block_recurrent_transformer import RecurrentAttentionEncoder

class RecurrentAttentionEncoderWithCls(RecurrentAttentionEncoder):
    def __init__(self,
                 dim: int,
                 dim_state: int,
                 dim_head: int = 64,
                 state_len: int = 512,
                 num_head: int = 8,
                 causal: bool = False,
                 num_layers: int = 5,
                 num_classes: int = 11) -> None:
        super().__init__(dim, dim_state, dim_head, state_len, num_head, causal, num_layers)
        self.num_classes = num_classes
        self.cls = nn.Linear(dim, num_classes)
        self.embedding = nn.Linear(num_classes, dim)
    
    def forward(self, feature, mask=None):
        if len(self.state) < 1:
            self.state = [None for _ in range(self.num_layers)]
        feature = torch.permute(feature, dims=[0, 2, 1])
        feature = self.embedding(feature)

        temp_state = []
        for att_block, state in zip(self.att_blocks, self.state):
            feature, state = att_block(feature, state=state)
            temp_state.append(state.detach())

        self.state = temp_state
        feature = feature * mask.transpose(1, 2)
        output = self.cls(feature).transpose(1, 2)
        return output

@AbstractBuildFactory.register('model')
class ASRFWithBRT(nn.Module):
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
        num_head=1,
        dim_head=1,
        sample_rate=1,
        state_len=512,
        num_layers=5,
        causal=False,
        num_stages_asb: Optional[int] = None,
        num_stages_brb: Optional[int] = None,
        out_feature: bool = False
    ) -> None:
        self.sample_rate = sample_rate
        if not isinstance(num_stages_asb, int):
            num_stages_asb = num_stages

        if not isinstance(num_stages_brb, int):
            num_stages_brb = num_stages

        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, num_features, 1)
        self.shared_layers = RecurrentAttentionEncoder(num_features, num_features, dim_head=dim_head,
                                                 state_len=state_len, num_head=num_head, causal=causal,
                                                 num_layers=num_layers)
        self.conv_cls = nn.Conv1d(num_features, num_classes, 1)
        self.conv_bound = nn.Conv1d(num_features, 1, 1)

        # action segmentation branch
        asb = [
            RecurrentAttentionEncoderWithCls(num_features, num_features, dim_head=dim_head,
                                        state_len=state_len, num_head=num_head, causal=causal,
                                        num_layers=num_layers, num_classes=num_classes)
            for _ in range(num_stages_asb - 1)
        ]

        # boundary regression branch
        brb = [
            RecurrentAttentionEncoderWithCls(num_features, num_features, dim_head=dim_head,
                                        state_len=state_len, num_head=num_head, causal=causal,
                                        num_layers=num_layers, num_classes=1)
            for _ in range(num_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.shared_layers._clear_memory_buffer()
        for asb in self.asb:
            asb._clear_memory_buffer()
            
        for brb in self.brb:
            brb._clear_memory_buffer()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask[:, :, ::self.sample_rate]
        
        out = self.conv_in(x)

        out = torch.permute(out, dims=[0, 2, 1]).contiguous()
        out = self.shared_layers(out, mask)

        out_feature = torch.permute(out, dims=[0, 2, 1]).contiguous()
        out_cls = self.conv_cls(out_feature)
        out_bound = self.conv_bound(out_feature)

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