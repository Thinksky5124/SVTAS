'''
Author       : Thyssen Wen
Date         : 2022-12-30 16:11:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-30 22:01:22
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/attention_layer.py
'''
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from .position_encoding import T5RelativePositionBias, RelativePosition
from ...utils import RotaryEmbedding

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# class

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

# activation functions

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class LaplacianAttnFn(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """

    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt(0.25 * math.pi)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

class MixedChunkAttentionLayer(nn.Module):
    """
    Mixed Chunk Attention Layer, implement by pytorch (ref:https://github.com/lucidrains/FLASH-pytorch),
    from paper <Transformer Quality in Linear Tim> :https://arxiv.org/pdf/2202.10447.pdf

    It also named Fast Linear Attention Single Head (FLASH).
    So, it is now only support single attention head.
    """
    def __init__(
        self,
        input_dim,
        group_size = 128,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        dropout = 0.,
        shift_tokens = False,
        laplace_attn_fn = True,
        reduce_group_non_causal_attn = False,
        position_encoding = False,
    ):
        super().__init__()
        hidden_dim = int(input_dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()

        # positional embeddings
        self.position_encoding = position_encoding
        if self.position_encoding:
            self.rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
            self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal = causal)
        else:
            self.rotary_pos_emb = None
            self.rel_pos_bias = None
        # norm

        self.norm = nn.InstanceNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)

        # whether to reduce groups in non causal linear attention

        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # projections

        self.to_hidden_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )

        self.to_hidden_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(input_dim, query_key_dim),
            nn.SiLU()
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)
        self.to_out = nn.Linear(hidden_dim, input_dim)
    
    def gen_key_padding_mask(self, masks):
        sum_masks = torch.sum(masks.squeeze(1), dim=-1) == 0.0
        for bs in range(sum_masks.shape[0]):
            if sum_masks[bs]:
                masks[bs] = 1.0
        return (masks == 0.0).squeeze(1)
        
    def forward(self, q, k, v, masks):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)

        q: [N C T]
        k: [N C T] not used
        v: [N C T]
        """
        # [N C T] -> [N T C]
        # prenorm
        q = self.norm(q)
        
        q = torch.transpose(q, 1, 2)
        v = torch.transpose(v, 1, 2)
        b, n, device, g = q.shape[0], q.shape[-2], q.device, self.group_size
        key_padding_mask = self.gen_key_padding_mask(masks=masks)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen
        if self.shift_tokens:
            # shfit q
            q_shift, q_pass = q.chunk(2, dim = -1)
            q_shift = F.pad(q_shift, (0, 0, 1, -1), value = 0.)
            q = torch.cat((q_shift, q_pass), dim = -1)
            # shfit v
            v_shift, v_pass = v.chunk(2, dim = -1)
            v_shift = F.pad(v_shift, (0, 0, 1, -1), value = 0.)
            v = torch.cat((v_shift, v_pass), dim = -1)

        # initial projections
        gate = self.to_hidden_gate(q)
        v = self.to_hidden_v(v)
        qk = self.to_qk(q)

        # offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # mask out linear attention keys
        if exists(key_padding_mask):
            lin_mask = rearrange(key_padding_mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups
        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))

            key_padding_mask = default(key_padding_mask, torch.ones((b, n), device = device, dtype = torch.bool))
            key_padding_mask = F.pad(key_padding_mask, (0, padding), value = False)

        # group along sequence
        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))

        if exists(key_padding_mask):
            key_padding_mask = rearrange(key_padding_mask, 'b (g j) -> b g 1 j', j = g)

        # calculate quadratic attention output
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        if exists(self.rel_pos_bias):
            sim = self.rel_pos_bias(sim)

        attn = self.attn_fn(sim)
        attn = self.dropout(attn)

        if exists(key_padding_mask):
            attn = attn.masked_fill(~key_padding_mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # calculate linear attention output
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)

            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
        else:
            context_einsum_eq = 'b d e' if self.reduce_group_non_causal_attn else 'b g d e'
            lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k, v) / n
            lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding
        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

        # gate
        out = gate * (quad_attn_out + lin_attn_out)

        # projection out
        out = self.to_out(out).transpose(1, 2)
        return out * masks

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads=1,
                 dropout=0.1,
                 batch_first=True):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.linears = self.clones(nn.Linear(embed_dim, embed_dim), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def clones(module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size = query.shape[0]
        len_q = query.shape[1]

        # merge key padding and attention masks
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, len_q).expand(-1, self.num_heads, -1, -1)

        # convert mask to float
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            mask = mask | attn_mask


        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)
        
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 causal=False) -> None:
        super().__init__()
        # self.att_layer = nn.MultiheadAttention(embed_dim=embed_dim,
        #                                        num_heads=num_heads,
        #                                        dropout=dropout,
        #                                        batch_first=True)
        self.att_layer = MultiHeadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.causal = causal
        self.num_heads = num_heads
    
    def gen_causal_mask(self, input_size, device):
        """
        Generates a causal mask of size (input_size, input_size) for attention
        """
        # [T, T]
        l_l_mask = torch.triu(torch.ones(input_size, input_size), diagonal=1) == 1
        l_l_mask = l_l_mask.to(device)
        return l_l_mask
    
    def gen_key_padding_mask(self, masks):
        sum_masks = torch.sum(masks.squeeze(1), dim=-1) == 0.0
        key_padding_mask = masks.detach().clone()
        for bs in range(sum_masks.shape[0]):
            if sum_masks[bs]:
                key_padding_mask[bs] = 1.0
        return (key_padding_mask == 0.0).squeeze(1)
    
    def forward(self, q, k, v, masks):
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        key_padding_mask = self.gen_key_padding_mask(masks=masks)
        if self.causal:
            causal_mask = self.gen_causal_mask(q.shape[1], masks)
        else:
            causal_mask = None
        
        out, att_map = self.att_layer(q, k, v, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
        out = torch.transpose(out, 1, 2)
        return out * masks

class MultiHeadChunkAttentionLayer(MultiHeadAttentionLayer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 chunck_size=1,
                 dropout=0,
                 causal=False) -> None:
        super().__init__(embed_dim, num_heads, dropout, causal)
        self.chunck_size = chunck_size
    
    def forward(self, q, k, v, masks):
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        chunck_masks = torch.transpose(masks, 1, 2)

        # chunck
        # padding for groups
        padding = padding_to_multiple_of(q.shape[1], self.chunck_size)
        temporal_size = q.shape[1]
        if padding > 0:
            q, k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (q, k, v))
            chunck_masks = F.pad(chunck_masks, (0, 0, 0, padding), value = 0.)
        g_size = q.shape[1] // self.chunck_size
        q, k, v, chunck_masks = map(lambda t: rearrange(t, 'b (g n) d -> (b g) n d', n = self.chunck_size), (q, k, v, chunck_masks))

        chunck_masks = torch.transpose(chunck_masks, 1, 2)
        key_padding_mask = self.gen_key_padding_mask(masks=chunck_masks)
        if self.causal:
            causal_mask = self.gen_causal_mask(q.shape[1], chunck_masks.device)
        else:
            causal_mask = None

        out, att_map = self.att_layer(q, k, v, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
        
        # gather group
        out = rearrange(out, '(b g) n d -> b (g n) d', g = g_size)
        out = torch.transpose(out, 1, 2)[:, :, :temporal_size]
        return out * masks

class MHRPRChunkAttentionLayer(MultiHeadChunkAttentionLayer):
    """
    Multi Head Realtive Position Representation Attention Layer (MHRPRAttentionLayer)
    implement by pytorch (ref:https://github.com/evelinehong/Transformer_Relative_Position_PyTorch),
    from paper <Self-Attention with Relative Position Representations> :https://arxiv.org/pdf/1803.02155.pdf
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 chunck_size=1,
                 causal=False):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        self.causal = causal
        self.chunck_size = chunck_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        
        self.fc_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([self.head_dim])))
        
    def relative_mulihead_attention_op(self, query, key, value, key_padding_mask=None, attn_mask=None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        # merge key padding and attention masks
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, len_q).expand(-1, self.num_heads, -1, -1)

        # convert mask to float
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            mask = mask | attn_mask

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.num_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.num_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = self.dropout(torch.softmax(attn, dim = -1))
        attn_map = attn

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.num_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.num_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.embed_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attn_map
    
    def forward(self, q, k, v, masks):
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        chunck_masks = torch.transpose(masks, 1, 2)

        # chunck
        # padding for groups
        padding = padding_to_multiple_of(q.shape[1], self.chunck_size)
        temporal_size = q.shape[1]
        if padding > 0:
            q, k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (q, k, v))
            chunck_masks = F.pad(chunck_masks, (0, 0, 0, padding), value = 0.)
        g_size = q.shape[1] // self.chunck_size
        q, k, v, chunck_masks = map(lambda t: rearrange(t, 'b (g n) d -> (b g) n d', n = self.chunck_size), (q, k, v, chunck_masks))

        chunck_masks = torch.transpose(chunck_masks, 1, 2)
        key_padding_mask = self.gen_key_padding_mask(masks=chunck_masks)
        if self.causal:
            causal_mask = self.gen_causal_mask(q.shape[1], chunck_masks.device)
        else:
            causal_mask = None

        out, att_map = self.relative_mulihead_attention_op(q, k, v, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
        
        # gather group
        out = rearrange(out, '(b g) n d -> b (g n) d', g = g_size)
        out = torch.transpose(out, 1, 2)[:, :, :temporal_size]
        return out * masks