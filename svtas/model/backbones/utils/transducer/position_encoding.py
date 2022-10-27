'''
Author       : Thyssen Wen
Date         : 2022-05-21 10:53:45
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-21 10:53:46
Description  : Transducer PositionEncoding Module ref:https://github.com/upskyy/Transformer-Transducer/blob/main/transformer_transducer/module.py
FilePath     : /ETESVS/model/backbones/utils/transducer/position_encoding.py
'''
from torch import Tensor
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implement the positional encoding (PE) function.
    PE_(pos, 2i)    =  sin(pos / 10000 ** (2i / d_model))
    PE_(pos, 2i+1)  =  cos(pos / 10000 ** (2i / d_model))
    """
    def __init__(
            self,
            d_model: int = 512,
            max_len: int = 5000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length, :]


class PositionWiseFeedForward(nn.Module):
    """
    Implement position-wise feed forward layer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(
            self,
            model_dim: int = 512,
            ff_dim: int = 2048,
            dropout: float = 0.1,
    ) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)