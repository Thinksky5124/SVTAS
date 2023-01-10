'''
Author       : Thyssen Wen
Date         : 2023-01-07 16:53:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-01-07 16:53:34
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/utils/attention_helper/position_encoding.py
'''
import math
import torch
import torch.nn as nn
from einops import rearrange
# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale + x

class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(SinusoidalPositionalEncoding, self).__init__()
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

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

        self.register_buffer(
            "position_ids",
            torch.arange(max_len).expand((1, -1)),
        )

    def forward(self, x):
        position_ids = self.position_ids[:, :x.shape[-1]]
        position_embeddings = self.pe(position_ids).transpose(1, 2)
        return x + position_embeddings
        
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings