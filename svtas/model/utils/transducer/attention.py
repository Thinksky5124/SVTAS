'''
Author       : Thyssen Wen
Date         : 2022-05-21 10:52:43
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-21 10:52:48
Description  : Transducer attention module ref:https://github.com/upskyy/Transformer-Transducer/blob/main/transformer_transducer/attention.py
FilePath     : /ETESVS/model/backbones/utils/transducer/attention.py
'''
from torch import Tensor
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all keys, divide each by sqrt(key_dim),
    and apply a softmax function to obtain the weights on the values
    Args: key_dim
        key_dim (int): dimension of key
    Inputs: query, key, value
        - **query** (batch, q_len, hidden_dim): tensor containing projection vector for decoder
        - **key** (batch, k_len, hidden_dim): tensor containing projection vector for encoder
        - **value** (batch, v_len, hidden_dim): value and key are the same
        - **mask** (batch, q_len, k_len): tensor containing mask vector for attn_distribution
    Returns: context, attn_distribution
        - **context** (batch, q_len, hidden_dim): tensor containing the context vector from attention mechanism
        - **attn_distribution** (batch, q_len, k_len): tensor containing the attention from the encoder outputs
    """
    def __init__(
            self,
            key_dim: int,
    ) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_key_dim = np.sqrt(key_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        key = key.transpose(1, 2)  # transpose 이후 key shape : (batch, enc_D << 1, enc_T)
        attn_distribution = torch.bmm(query, key) / self.sqrt_key_dim

        if mask is not None:
            attn_distribution = attn_distribution.masked_fill(mask, -np.inf)

        attn_distribution = F.softmax(attn_distribution, dim=-1)  # (batch, dec_T, enc_T)
        context = torch.bmm(attn_distribution, value)  # context shape : (batch, dec_T, enc_D << 1)

        return context, attn_distribution


class MultiHeadAttention(nn.Module):
    """
    This technique is proposed in this paper. https://arxiv.org/abs/1706.03762
    Perform the scaled dot-product attention in parallel.
    Args:
        model_dim (int): the number of features in the multi-head attention (default : 512)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
    Inputs: query, key, value, mask
        - **query** (batch, q_len, hidden_dim): tensor containing projection vector for encoder
        - **key** (batch, k_len, hidden_dim): tensor containing projection vector for encoder
        - **value** (batch, v_len, hidden_dim): tensor containing projection vector for encoder
        - **mask** (batch, q_len, k_len): tensor containing mask vector for self attention distribution
    Returns: context, attn_distribution
        - **context** (batch, dec_len, dec_hidden): tensor containing the context vector from attention mechanism
        - **attn_distribution** (batch, dec_len, enc_len): tensor containing the attention from the encoder outputs
    """
    def __init__(
            self,
            model_dim: int,
            num_heads: int,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scaled_dot = ScaledDotProductAttention(self.head_dim)
        self.query_fc = nn.Linear(model_dim, num_heads * self.head_dim)
        self.key_fc = nn.Linear(model_dim, num_heads * self.head_dim)
        self.value_fc = nn.Linear(model_dim, num_heads * self.head_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch = query.size(0)

        query = self.query_fc(query).view(batch, -1, self.num_heads, self.head_dim)
        key = self.key_fc(key).view(batch, -1, self.num_heads, self.head_dim)
        value = self.value_fc(value).view(batch, -1, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3).contiguous().view(batch * self.num_heads, -1, self.head_dim)
        key = key.permute(0, 2, 1, 3).contiguous().view(batch * self.num_heads, -1, self.head_dim)
        value = value.permute(0, 2, 1, 3).contiguous().view(batch * self.num_heads, -1, self.head_dim)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        context, attn_distribution = self.scaled_dot(query, key, value, mask)

        context = context.view(batch, self.num_heads, -1, self.head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.num_heads * self.head_dim)  # (B, T, D)

        return context, attn_distribution