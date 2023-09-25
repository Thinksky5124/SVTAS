'''
Author       : Thyssen Wen
Date         : 2023-02-25 15:33:51
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-28 08:49:52
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/block_recurrent_transformer/block_recurrent_transformer.py
'''
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from ....builder import HEADS
from .helper_function import (default, exists, apply_rotary_pos_emb, cast_tuple,
                              RMSNorm, RotaryEmbedding, FeedForward)


class RecurrentStateGate(nn.Module):
    """Poor man's LSTM
    """

    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x, state):
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)


class Attention(nn.Module):
    """Shamelessly copied from github.com/lucidrains/RETRO-pytorch
    """
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        null_kv = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # allowing for attending to nothing (null function)
        # and to save attention from breaking if all retrieved chunks are padded out
        self.null_kv = nn.Parameter(torch.randn(2, inner_dim)) if null_kv else None

    def forward(self, x, mask = None, context = None, pos_emb = None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        x = self.norm(x)
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * scale

        # apply relative positional encoding (rotary embeddings)
        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # add null key / values
        if exists(self.null_kv):
            nk, nv = self.null_kv.unbind(dim = 0)
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h = h), (nk, nv))
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        # derive query key similarities
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            if exists(self.null_kv):
                mask = F.pad(mask, (1, 0), value = True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # combine heads linear out
        return self.to_out(out)


class RecurrentAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_state: int,
        dim_head: int = 64,
        state_len: int = 512,
        heads: int = 8,
        causal: bool = False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.dim = dim
        self.dim_state = dim_state

        self.heads = heads
        self.causal = causal
        self.state_len = state_len
        rotary_emb_dim = max(dim_head // 2, 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        
        self.input_self_attn = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
        self.state_self_attn = Attention(dim_state, heads = heads, causal = causal, **attn_kwargs)

        self.input_state_cross_attn = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
        self.state_input_cross_attn = Attention(dim_state, heads = heads, causal = causal, **attn_kwargs)

        self.proj_gate = RecurrentStateGate(dim)
        self.ff_gate = RecurrentStateGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias = False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias = False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)


    def forward(
        self,
        x,
        state= None,
        mask = None,
        state_mask = None
    ):
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = torch.zeros((batch, self.state_len, self.dim_state), device=device)
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device = device)
        state_pos_emb = self.rotary_pos_emb(self.state_len, device = device)
        input_attn = self.input_self_attn(x, mask = mask, pos_emb = self_attn_pos_emb)
        state_attn = self.state_self_attn(state, mask = state_mask, pos_emb = state_pos_emb)

        # TODO: This is different from how it is implemented in the paper, because the Keys and Values aren't shared
        # between the cross attention and self-attention. I'll implement that later, this is faster for now.
        input_as_q_cross_attn = self.input_state_cross_attn(x, context = state, mask = mask)
        state_as_q_cross_attn = self.state_input_cross_attn(state, context = x, mask = state_mask)

        projected_input = self.input_proj(torch.concat((input_as_q_cross_attn, input_attn), dim=2))
        projected_state = self.state_proj(torch.concat((state_as_q_cross_attn, state_attn), dim=2))

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state

class RecurrentAttentionEncoder(nn.Module):
    def __init__(self,
                 dim: int,
                 dim_state: int,
                 dim_head: int = 64,
                 state_len: int = 512,
                 num_head: int = 8,
                 causal: bool = False,
                 num_layers:int = 5) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.att_blocks = nn.ModuleList([RecurrentAttentionBlock(dim, dim_state, dim_head=dim_head,
                                                                  state_len=state_len, heads=num_head, causal=causal)
                                                                  for _ in range(num_layers)])
        self.state = []
    
    def _clear_memory_buffer(self):
        self.state = []
    
    def forward(self, feature, mask=None):
        if len(self.state) < 1:
            self.state = [None for _ in range(self.num_layers)]
        
        temp_state = []
        for att_block, state in zip(self.att_blocks, self.state):
            feature, state = att_block(feature, state=state)
            temp_state.append(state.detach())

        self.state = temp_state

        return feature * mask.transpose(1, 2)
    
@AbstractBuildFactory.register('model')
class BRTClassificationHead(nn.Module):
    """Block Recurrent Transformer
    paper: https://arxiv.org/pdf/2203.07852
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes,
                 num_head=1,
                 dim_head=1,
                 sample_rate=1,
                 state_len=512,
                 num_layers=5,
                 causal=False,
                 out_feature=False):
        super(BRTClassificationHead, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.encoder = RecurrentAttentionEncoder(hidden_channels, hidden_channels, dim_head=dim_head,
                                                 state_len=state_len, num_head=num_head, causal=causal,
                                                 num_layers=num_layers)
        if in_channels != hidden_channels:
            self.embedding = nn.Conv1d(in_channels, hidden_channels, 1)
        else:
            self.embedding = None
        self.cls = nn.Conv1d(hidden_channels, num_classes, 1)

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        self.encoder._clear_memory_buffer()

    def forward(self, x, mask):
        mask = mask[:, :, ::self.sample_rate]
        if self.embedding is not None:
            x = self.embedding(x)

        x = torch.permute(x, dims=[0, 2, 1]).contiguous()
        feature = self.encoder(x, mask=mask)

        feature = torch.permute(feature, dims=[0, 2, 1]).contiguous()

        out = self.cls(feature)
        outputs = out.unsqueeze(0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        
        if self.out_feature is True:
            return feature, outputs
        return outputs