'''
Author       : Thyssen Wen
Date         : 2022-10-28 15:44:56
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-30 16:36:49
Description  : MViT ref:https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/video_model_builder.py
FilePath     : /SVTAS/svtas/model/backbones/video/mvit.py
'''

import torch
from torch import nn
import math
from functools import partial
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from ....utils.logger import get_logger
from svtas.utils import AbstractBuildFactory
from mmcv.runner import load_state_dict
from collections import OrderedDict
import re
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper

from .reversible_mvit import ReversibleMViT

from ..utils import (TwoStreamFusion, calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed, PatchEmbed,
    round_width, MultiScaleBlock)


@AbstractBuildFactory.register('model')
class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.
    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self,
                 pool_first = False,
                 spatial_size = 224,
                 input_channel_num = [3],
                 use_2d_patch = False,
                 enable_rev = False,
                 patch_stride = [2, 4, 4],
                 clip_seg_num = 32,
                 embed_dim = 96,
                 num_heads = 1,
                 mlp_ratio = 4.0,
                 qkv_bias = True,
                 drop_rate = 0.0,
                 depth = 24,
                 drop_path_rate = 0.3,
                 mode = "conv",
                 cls_embed_on = True,
                 use_mean_pooling = True,
                 use_abs_pos = False,
                 use_fixed_sincos_pos = False,
                 sep_pos_embed = True,
                 rel_pos_spatial = True,
                 rel_pos_temporal = True,
                 norm = "layernorm",
                 act_checkpoint = False,
                 patch_kernel = (3, 7, 7),
                 pathch_padding = (1, 3, 3),
                 cfg_dim_mul = [[2, 2.0], [5, 2.0], [21, 2.0]],
                 cfg_head_mul = [[2, 2.0], [5, 2.0], [21, 2.0]],
                 pool_q_stride = [[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 2, 2], [6, 1, 1, 1], [7, 1, 1, 1],
                                  [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1],
                                  [16, 1, 1, 1], [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1], [20, 1, 1, 1], [21, 1, 2, 2], [22, 1, 1, 1], [23, 1, 1, 1]],
                 pool_kvq_kernel = [3, 3, 3],
                 pool_kv_stride_adaptive = [1, 8, 8],
                 pool_kv_stride = [],
                 norm_stem = False,
                 rel_pos_zero_init = False,
                 residual_pooling = False,
                 dim_mul_in_att = True,
                 separate_qkv = False,
                 rev_respath_fuse = "concat",
                 zero_decay_pos_cls=False,
                 res_q_fusion = "concat",
                 buffer_layers = [1,3, 14],
                 pretrained=None):
        super().__init__()
        # Prepare input.
        self.pretrained = pretrained
        in_chans = input_channel_num[0]
        self.zero_decay_pos_cls = zero_decay_pos_cls
        self.use_2d_patch = use_2d_patch
        self.enable_rev = enable_rev
        self.patch_stride = patch_stride
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = clip_seg_num // self.patch_stride[0]
        self.H = spatial_size // self.patch_stride[1]
        self.W = spatial_size // self.patch_stride[2]
        # Prepare output.
        # Prepare backbone
        self.drop_rate = drop_rate
        self.cls_embed_on = cls_embed_on
        self.use_mean_pooling = use_mean_pooling
        # Params for positional embedding
        self.use_abs_pos = use_abs_pos
        self.use_fixed_sincos_pos = use_fixed_sincos_pos
        self.sep_pos_embed = sep_pos_embed
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        self.rev_respath_fuse = rev_respath_fuse
        if norm == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=patch_kernel,
            stride=patch_stride,
            padding=pathch_padding,
            conv_2d=self.use_2d_patch,
        )

        if act_checkpoint:
            self.patch_embed = checkpoint_wrapper.checkpoint_wrapper(self.patch_embed)
        self.input_dims = [clip_seg_num, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg_dim_mul)):
            dim_mul[cfg_dim_mul[i][0]] = cfg_dim_mul[i][1]
        for i in range(len(cfg_head_mul)):
            head_mul[cfg_head_mul[i][0]] = cfg_head_mul[i][1]

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        for i in range(len(pool_q_stride)):
            stride_q[pool_q_stride[i][0]] = pool_q_stride[i][
                1:
            ]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if pool_kv_stride_adaptive is not None:
            _stride_kv = pool_kv_stride_adaptive
            pool_kv_kernel = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_kernel.append([i] + _stride_kv)

        for i in range(len(pool_kv_stride)):
            stride_kv[pool_kv_stride[i][0]] = pool_kv_stride[
                i
            ][1:]
            if pool_kvq_kernel is not None:
                pool_kv[
                    pool_kv_stride[i][0]
                ] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in pool_kv_stride[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if norm_stem else None

        input_size = self.patch_dims

        if self.enable_rev:

            # rev does not allow cls token
            assert not self.cls_embed_on

            self.rev_backbone = ReversibleMViT(embed_dim,
                                               depth,
                                               num_heads,
                                               mlp_ratio,
                                               qkv_bias,
                                               drop_path_rate,
                                               drop_rate,
                                               res_q_fusion,
                                               norm,
                                               cfg_dim_mul,
                                               cfg_head_mul,
                                               buffer_layers,
                                               cls_embed_on,
                                               mode,
                                               pool_first,
                                               rel_pos_spatial,
                                               rel_pos_temporal,
                                               rel_pos_zero_init,
                                               residual_pooling,
                                               separate_qkv,
                                               self)

            embed_dim = round_width(
                embed_dim, dim_mul.prod(), divisor=num_heads
            )

            self.fuse = TwoStreamFusion(
                rev_respath_fuse, dim=2 * embed_dim
            )

            if "concat" in self.rev_respath_fuse:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:

            self.blocks = nn.ModuleList()

            for i in range(depth):
                num_heads = round_width(num_heads, head_mul[i])
                if dim_mul_in_att:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i],
                        divisor=round_width(num_heads, head_mul[i]),
                    )
                else:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i + 1],
                        divisor=round_width(num_heads, head_mul[i + 1]),
                    )
                attention_block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                    pool_first=pool_first,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                    rel_pos_zero_init=rel_pos_zero_init,
                    residual_pooling=residual_pooling,
                    dim_mul_in_att=dim_mul_in_att,
                    separate_qkv=separate_qkv,
                )

                if act_checkpoint:
                    attention_block = checkpoint_wrapper.checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(num_frames=clip_seg_num,
                                                                      patch_stride=patch_stride,
                                                                      crop_size=spatial_size,
                                                                      depth=depth,
                                                                      pool_q_stride=pool_q_stride)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        return x
    
    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'backbone.', r'')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                def revise_keys_fn(state_dict, revise_keys=[(r'module.', r'')]):
                    # strip prefix of state_dict
                    metadata = getattr(state_dict, '_metadata', OrderedDict())
                    for p, r in revise_keys:
                        state_dict = OrderedDict(
                            {re.sub(p, r, k): v
                            for k, v in state_dict.items()})
                    # Keep metadata in state_dict
                    state_dict._metadata = metadata
                    return state_dict

                logger  = get_logger("SVTAS")
                checkpoint = torch.load(self.pretrained)
                load_state_dict(self, checkpoint['model_state'], strict=False, logger=logger)
            else:
                self.apply(self._init_weights)
        else:
            self.apply(self._init_weights)

    def forward(self, x, masks):
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        if self.enable_rev:
            x = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, thw = blk(x, thw)

            if self.use_mean_pooling:
                if self.cls_embed_on:
                    x = x[:, 1:]
                x = x.mean(1)
                x = self.norm(x)
            elif self.cls_embed_on:
                x = self.norm(x)
                x = x[:, 0]
            else:  # this is default, [norm->mean]
                x = self.norm(x)
                x = x.mean(1)

        return x