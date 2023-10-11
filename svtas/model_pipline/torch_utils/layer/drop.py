'''
Author       : Thyssen Wen
Date         : 2023-10-11 18:00:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 18:01:32
Description  : ref: https://github.com/open-mmlab/mmcv/blob/57c4e25e06e2d4f8a9357c84bcd24089a284dc88/mmcv/cnn/bricks/drop.py
FilePath     : /SVTAS/svtas/model_pipline/torch_utils/layer/drop.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from svtas.utils import AbstractBuildFactory


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@AbstractBuildFactory.register('model')
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)