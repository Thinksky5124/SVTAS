'''
Author       : Thyssen Wen
Date         : 2023-01-07 16:52:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-15 15:57:11
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/utils/attention_helper/attention_layer.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from .position_encoding import T5RelativePositionBias, RelativePosition
from ...utils import RotaryEmbedding
from timm.models.layers import DropPath, trunc_normal_

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
        reduce_group_non_causal_attn = True,
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
        # if exists(self.rel_pos_bias):
        #     sim = self.rel_pos_bias(sim)

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

class GAU(nn.Module):
    def __init__(
        self,
        embed_dim = 128,
        expansion_factor = 2.,
        dropout = 0.,
        laplace_attn_fn = False,
        position_encoding=True
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()

        self.to_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU()
        )

        self.to_v = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(embed_dim, heads = 2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # positional embeddings
        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pos_enc = RotaryEmbedding(dim = min(32, embed_dim))
        else:
            self.pos_enc = None

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, seq_len, device = query.shape[0], query.shape[-2], query.device

        # merge key padding and attention masks
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, seq_len)

        # convert mask to float
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            mask = mask | attn_mask
            
        if self.position_encoding:
            query = self.pos_enc.rotate_queries_or_keys(query)

        normed_x = self.norm(query)
        gate = self.to_gate(normed_x)
        v = self.to_gate(value)

        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)

        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len

        attn = self.attn_fn(sim)
        att_map = attn
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(mask, 0.)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        return out, att_map

class GAUAttentionLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 dropout=0.0,
                 causal=False,
                 position_encoding=True,
                 num_gau=1) -> None:
        super().__init__()
        self.att_layers = nn.ModuleList([GAU(embed_dim=embed_dim,
                                            position_encoding=position_encoding) for _ in range(num_gau)])
        self.causal = causal
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
    
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
        
        for layer in self.att_layers:
            out, att_map = layer(q, k, v, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
            q = out
        out = torch.transpose(out, 1, 2)
        if self.dropout is not None:
            out = self.dropout(out)
        return out * masks

class GAUAChunkttentionLayer(GAUAttentionLayer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 chunck_size=1,
                 dropout=0,
                 causal=False,
                 position_encoding=True) -> None:
        super().__init__(embed_dim, num_heads, dropout, causal, position_encoding)
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

        for layer in self.att_layers:
            out, att_map = layer(q, k, v, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
            q = out
        
        # gather group
        out = rearrange(out, '(b g) n d -> b (g n) d', g = g_size)
        out = torch.transpose(out, 1, 2)[:, :, :temporal_size]
        if self.dropout is not None:
            out = self.dropout(out)
        return out * masks

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads=1,
                 position_encoding=True):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])

        # positional embeddings
        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pos_enc = RotaryEmbedding(dim = min(32, embed_dim))
        else:
            self.pos_enc = None
    
    @staticmethod
    def attention(query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        p_attn = F.softmax(scores, dim = -1)
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


        query, key, value = [l(x) for l, x in zip(self.embed_layers, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        if self.position_encoding:
            query = self.pos_enc.rotate_queries_or_keys(query)
            key = self.pos_enc.rotate_queries_or_keys(key)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return x, attn
        
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 causal=False,
                 position_encoding=True) -> None:
        super().__init__()
        self.att_layer = MultiHeadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            position_encoding=position_encoding)
        self.causal = causal
        self.num_heads = num_heads
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
    
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
        if self.dropout is not None:
            out = self.dropout(out)
        return out * masks

class MultiHeadChunkAttentionLayer(MultiHeadAttentionLayer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 chunck_size=1,
                 dropout=0,
                 causal=False,
                 position_encoding=True) -> None:
        super().__init__(embed_dim, num_heads, dropout, causal, position_encoding)
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
        if self.dropout is not None:
            out = self.dropout(out)
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
                 causal=False,
                 position_encoding=False):
        super().__init__(embed_dim, num_heads, dropout, causal, position_encoding)
        
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

def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
        return conv
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1/dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin

class LinearMultiHeadAttention(nn.Module):
    def __init__(self,
                 chunk_size,
                 embed_dim,
                 num_heads=1,
                 position_encoding=True) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])

        # positional embeddings
        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pos_enc = RotaryEmbedding(dim = min(32, embed_dim))
        else:
            self.pos_enc = None

        self.dropout = nn.Dropout()
        self.P_bar = None
        self.E = get_EF(chunk_size, embed_dim, "learnable", embed_dim)
        self.F = get_EF(chunk_size, embed_dim, "learnable", embed_dim)
        self.is_proj_tensor = isinstance(self.E, torch.Tensor)
    
    @staticmethod
    def attention(query, key, value, E, F, mask=None, is_proj_tensor=False):
        """
        ref: https://github.com/tatp22/linformer-pytorch/blob/master/linformer_pytorch/linformer_pytorch.py
        """

        key = key.transpose(-1,-2)
        if is_proj_tensor:
            E = E.to(key.device)
            key = torch.matmul(key, E)
        else:
            key = E(key)

        query = torch.matmul(query, key)

        P_bar = query/torch.sqrt(torch.tensor(query.shape[-1]).type(query.type())).to(query.device)
        if mask is not None:
            mask = mask.to(query.device)
            P_bar = P_bar.masked_fill_(mask, float('-inf'))
        P_bar = P_bar.softmax(dim=-2)

        value = value.transpose(-1,-2)
        if is_proj_tensor:
            F = F.to(value.device)
            value = torch.matmul(value, F)
        else:
            value = F(value)

        value = value.transpose(-1,-2)
        attn = torch.matmul(P_bar, value)
        return attn, P_bar
    
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


        query, key, value = [l(x) for l, x in zip(self.embed_layers, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        if self.position_encoding:
            query = self.pos_enc.rotate_queries_or_keys(query)
            key = self.pos_enc.rotate_queries_or_keys(key)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, self.E, self.F, mask=mask.transpose(-1, -2), is_proj_tensor=self.is_proj_tensor)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return x, attn

class LinearAttentionLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 chunk_size,
                 num_heads,
                 dropout=0.0,
                 causal=False,
                 position_encoding=True) -> None:
        super().__init__()
        self.att_layer = LinearMultiHeadAttention(chunk_size=chunk_size,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         position_encoding=position_encoding)
        self.causal = causal
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
    
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
        if self.dropout is not None:
            out = self.dropout(out)
        return out * masks

class LinearChunkAttentionLayer(LinearAttentionLayer):
    def __init__(self,
                 embed_dim,
                 num_heads=1,
                 chunck_size=1,
                 dropout=0,
                 causal=False,
                 position_encoding=True) -> None:
        super().__init__(embed_dim, chunck_size, num_heads, dropout, causal, position_encoding)
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
        if self.dropout is not None:
            out = self.dropout(out)
        return out * masks