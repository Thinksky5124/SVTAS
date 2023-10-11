'''
Author       : Thyssen Wen
Date         : 2023-02-13 15:43:44
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-25 18:36:24
Description  : Swin Transformer With Stochastic Backpropagation ref:https://github.com/amazon-science/stochastic-backpropagation.git
FilePath     : /SVTAS/svtas/model/backbones/video/swin_transformer_3d_sbp.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

from svtas.model_pipline.torch_utils import load_checkpoint, DropPath, trunc_normal_
from ....utils.logger import get_logger
from svtas.utils import AbstractBuildFactory
from .swin_transformer_3d import get_window_size, window_partition, window_reverse, compute_mask, PatchEmbed3D, PatchMerging

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

def assert_grad_mask(grad_mask, downsample):
    """make sure the grad_mask and downsample rate is matching.

    Args:
        grad_mask (torch.Tensor): any shape.
        downsample (int): downsample rate.

    """
    if grad_mask is not None:
        assert (
            downsample == grad_mask.numel() / grad_mask.sum()
        ), f"downsample and grad_mask must match. downsample:{downsample}, grad_mask_downsample: {grad_mask.numel() / grad_mask.sum()}"


class SBPmlp(object):
    # todo: merge this class into SBPMlpFunc.
    @staticmethod
    def forward(grad_mask, downsample, forward_fn, x, *params):
        """
        Args:
          grad_mask (torch.Tensor[B,D]): torch.bool. True means kept and False means dropt.
          downsample (int): The number to downsample along the temporal dimension.
          forward_fn (callable): The forward function in MLP.
          x (torch.Tensor[B,D,H,W,C]): input to MLP.
          params (tuple): The learnable parameters in MLP.

        Returns: tuple.
          (output, tensors_to_save).
          - output: shape [B,D,H,W,C]. The output of mlp.
          - tensors_to_save (tuple): The tensors needed to be saved for backward.
        """
        B, D, H, W, C = x.shape
        Dd = D // downsample

        # forward
        with torch.no_grad():
            y, random_tensor = forward_fn(x)

        # save for backward
        x_w_grad = x.masked_select(grad_mask.view(B, D, 1, 1, 1)).view(B, Dd, H, W, C)
        tensors_to_save = (
            forward_fn,
            x_w_grad,
            params,
            random_tensor,
        )

        return y, tensors_to_save

    @staticmethod
    def backward(saved_tensors, dy):
        """
        Args:
          dy (torch.Tensor[B,Dd,H,W,C]): gradient of output. zeros are removed.
        """

        (
            forward_fn,
            x_w_grad,
            params,
            random_tensor,
        ) = saved_tensors

        # detach x_w_grad
        x_w_grad = x_w_grad.detach().requires_grad_(True)

        with torch.enable_grad():
            y, random_tensor = forward_fn(x_w_grad, random_tensor)
        input_grads = torch.autograd.grad(y, (x_w_grad,) + params, dy)
        return (None, None, None) + input_grads


class SBPMlpFunc(torch.autograd.Function):
    """Mlp Function with stochastic backpropagation"""

    @staticmethod
    def forward(ctx, grad_mask, downsample, forward_fn, x, *params):
        ctx.grad_mask = grad_mask
        ctx.downsample = downsample
        assert_grad_mask(grad_mask, downsample)
        y, tensors_to_save = SBPmlp.forward(
            grad_mask, downsample, forward_fn, x, *params
        )
        ctx.tensors = tensors_to_save
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Args:
          dy (torch.Tensor[B,D,H,W,C]): gradient of output. zeros are removed.
        """
        B, D, H, W, C = dy.shape
        grad_mask = ctx.grad_mask
        Dd = D // ctx.downsample
        dy = dy.masked_select(grad_mask.view(B, D, 1, 1, 1)).view(B, Dd, H, W, C)

        grads = list(SBPmlp.backward(ctx.tensors, dy))
        dx = torch.zeros(B, D, H, W, C, dtype=dy.dtype, device=dy.device)
        dx.masked_scatter_(grad_mask.view(B, D, 1, 1, 1), grads[3])
        grads[3] = dx
        del ctx.tensors, ctx.grad_mask, ctx.downsample
        return tuple(grads)


class PFDotProductAttention(torch.autograd.Function):
    """
    y = softmax(q @ k.T + bias) @ v

    drop q, A but not k, v
    """

    @staticmethod
    def run_fn(q, k, v, bias, scale, mask):
        Nq = q.shape[2]
        B_, nH, Nk, C = k.shape

        q = q * scale
        A = q @ k.transpose(-2, -1) + bias

        if mask is not None:
            nW = mask.shape[0]
            A = A.view(B_ // nW, nW, nH, Nq, Nk) + mask.unsqueeze(1).unsqueeze(0)
            A = A.view(B_, nH, Nq, Nk)

        A = torch.softmax(A, dim=-1)  # shape: [B,nH,Nq,Nk]
        y = A @ v
        return y

    @staticmethod
    def forward(ctx, q, k, v, bias, qk_scale, mask=None, grad_mask=None, downsample=1):
        """forward m-a

        Args:
            ctx (context object): context object.
            q (torch.Tensor[B_,nH,N,C]): query. N = window_size. B_ = B * num_windows
            k (torch.Tensor[B_,nH,N,C]): key.
            v (torch.Tensor[B_,nH,N,C]): value.
            bias (torch.Tensor[nH,N,N]): position bias.
            mask (torch.Tensor[num_windows, N, N]): Attention mask.
            grad_mask (torch.Tensor[B_,N]): torch.bool. True means kept and False means dropt.
            downsample (int): The number to downsample along the temporal dimension.

        Returns: torch.Tensor[B_,nH,N,C]. attention.

        """
        assert_grad_mask(grad_mask, downsample)
        B_, nH, N, C = q.shape
        scale = qk_scale
        with torch.no_grad():
            y = PFDotProductAttention.run_fn(q, k, v, bias, scale, mask)

        if downsample == 1:
            ctx.tensors = (q, k, v, bias, scale, mask)
        else:
            Nd = N // downsample
            indices = torch.arange(0, N, device=q.device)
            indices = indices.masked_select(grad_mask[0]).tolist()
            q = q[:, :, indices, :].contiguous()  # [B_, nH, Nd, C]
            bias = bias[:, indices, :].contiguous()  # [nH, Nd, N]
            if mask is not None:
                # mask = mask.permute(1, 0, 2)[indices].permute(1, 0, 2).contiguous()
                mask = mask[:, indices, :].contiguous()
                # mask = mask.masked_select(grad_mask[:, None, :, None]).view(nH, Nd, N)

            ctx.tensors = (q, k, v, bias, scale, mask)
            ctx.indices = indices
        ctx.grad_mask = grad_mask
        ctx.downsample = downsample
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """
        Args:
            dy (torch.Tensor[B_,nH,N,C])
        """

        # shape:
        # Nd = N //downsample
        # A: [B_,nH,Nd,N]
        # q: [B_,nH,Nd,C]
        # k, v: [B_,nH,N,C]
        downsample = ctx.downsample
        grad_mask = ctx.grad_mask
        q, k, v, bias, scale, mask = ctx.tensors
        B_, nH, Nd, C = q.shape
        N = Nd * downsample

        if downsample != 1:
            indices = ctx.indices
            # dy = dy.permute(2, 0, 1, 3)[indices].permute(1, 2, 0, 3).contiguous()
            dy = dy.masked_select(grad_mask[:, None, :, None]).view(B_, nH, Nd, C)

        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            bias = bias.detach().requires_grad_(True)
            y = PFDotProductAttention.run_fn(q, k, v, bias, scale, mask)
        dq, dk, dv, d_bias = torch.autograd.grad(y, (q, k, v, bias), dy)

        # upsample back
        if downsample != 1:
            B_, nH, N, C = k.shape
            dq = torch.zeros(
                B_, nH, N, C, dtype=dq.dtype, device=dq.device
            ).masked_scatter_(grad_mask[:, None, :, None], dq)

            d_bias1 = torch.zeros(N, nH, N, dtype=d_bias.dtype, device=d_bias.device)
            d_bias1[indices] = d_bias.permute(1, 0, 2)
            d_bias1 = d_bias1.permute(1, 0, 2).contiguous()
            d_bias = d_bias1

            # d_bias1 = torch.zeros(
            #     B_, nH, N, N, dtype=d_bias.dtype, device=d_bias.device
            # )
            # d_bias1.masked_scatter_(grad_mask[:, None, :, None], d_bias)
            # d_bias1 = d_bias1.sum(0)
            # d_bias = d_bias1

        del ctx.tensors, ctx.grad_mask
        return dq, dk, dv, d_bias, None, None, None, None
        
def drop_path_op(x, random_tensor=None, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x, random_tensor
    keep_prob = 1 - drop_prob

    if random_tensor is None:
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

    output = x.div(keep_prob) * random_tensor
    return output, random_tensor

class Mlp(nn.Module):

    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.norm = norm_layer(in_features)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.drop_path_prob = drop_path

    def forward_fn(self, x, droppath_random_tensor=None):
        shortcut = x
        # norm
        x = self.norm(x)

        # mlp
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # drop_path
        x, random_tensor = drop_path_op(
            x, droppath_random_tensor, self.drop_path_prob, training=self.training
        )
        return shortcut + x, random_tensor

    def forward(self, x):
        return self.forward_fn(x)


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w)
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # self.qkv_op = AttenQKVProj.apply

        self.pf_dot_prod_attn_op = PFDotProductAttention.apply

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition_grad_mask(self, grad_mask, orig_x_shape):
        """generate grad_mask for windowed attention

        Args:
            grad_mask (torch.Tensor[B,D]): the input grad_mask
            unpartitioned_x_shape (List): (B,D,H,W,C)

        Returns: torch.Tensor[B_, N]: the grad_mask for attention. B_ = B x num_windows, N = window_d x window_h x window_w

        """
        if grad_mask is None:
            return None
        B0, D0 = grad_mask.shape
        B, D, H, W, C = orig_x_shape
        # assert B == B0 and D == D0, "grad_mask and orig_x_shape shape mismatch"
        assert D == D0, "grad_mask and orig_x_shape shape mismatch"
        window_d, window_h, window_w = self.window_size
        grad_mask = torch.tile(grad_mask[:, :, None, None], (1, 1, H, W))  # [B,D,H,W]
        grad_mask = grad_mask.reshape(
            B, D // window_d, window_d, H // window_h, window_h, W // window_w, window_w
        )
        grad_mask = grad_mask.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            -1, reduce(mul, self.window_size)
        )
        return grad_mask

    def forward(self, x, orig_x_shape, mask=None, grad_mask=None, downsample=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            orig_x_shape (List): [B,D,H,W,C]
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        grad_mask_window = self.window_partition_grad_mask(grad_mask, orig_x_shape)

        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(
            N, N, -1
        )  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww

        # # orig attn
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
        #         1
        #     ).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        #     attn = self.softmax(attn)

        # # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # sbp attn
        x = self.pf_dot_prod_attn_op(
            q,
            k,
            v,
            relative_position_bias,
            self.scale,
            mask,
            grad_mask_window,
            downsample,
        )
        x = x.transpose(1, 2).reshape(B_, N, C)

        # x = self.qkv_op(
        #     x, self.proj.weight, self.proj.bias, grad_mask_window, downsample
        # )

        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        with_gd=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.with_gd = with_gd

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path_layer = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            drop_path=drop_path,
        )
        self.mlp_op = SBPMlpFunc.apply

    def forward_part1(self, x, mask_matrix, grad_mask=None, downsample=None):
        """
        Args:
            grad_mask (torch.Tensor[B,D]): The grad mask.
        """
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
            if grad_mask is not None:
                shifted_grad_mask = torch.roll(
                    grad_mask, shifts=-shift_size[0], dims=1
                )  # shift along dimention D.
            else:
                shifted_grad_mask = grad_mask
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_grad_mask = grad_mask
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows,
            orig_x_shape=x.shape,
            mask=attn_mask,
            grad_mask=shifted_grad_mask,
            downsample=downsample,
        )  # B*nW, Wd*Wh*Ww, C
        # attn_windows = x_windows * 2

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(
            attn_windows, window_size, B, Dp, Hp, Wp
        )  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward(self, x, mask_matrix, grad_mask=None, downsample=None):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
            grad_mask (torch.Tensor[B,D]): mask for grad drop.
            downsample (int): 1 / keep rate.
        """
        if not self.with_gd:
            grad_mask = None
            downsample = 1
        if downsample == 1:
            grad_mask = None

        # forward attention
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(
                self.forward_part1, x, mask_matrix, grad_mask, downsample
            )
        else:
            x = self.forward_part1(x, mask_matrix, grad_mask, downsample)
        x = shortcut + self.drop_path_layer(x)

        # forward MLP
        if downsample != 1:
            x = self.mlp_op(
                grad_mask,
                downsample,
                self.mlp.forward_fn,
                x,
                *tuple(self.mlp.parameters()),
            )
        else:
            if self.use_checkpoint:
                x, _ = checkpoint.checkpoint(self.mlp, x)
            else:
                x, _ = self.mlp(x)

        def hook_fn(dx):
            return dx * grad_mask[:, :, None, None, None]

        if self.training and downsample != 1:
            x.register_hook(hook_fn)

        return x
        
class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        with_gd=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if with_gd is None:
            with_gd = [0] * depth
        assert len(with_gd) == depth, "length of `with_gd` should be the same as depth"

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    with_gd=with_gd[i],
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, grad_mask, gd_downsample):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for i, blk in enumerate(self.blocks):
            # print(f'    layer {i}')
            x = blk(x, attn_mask, grad_mask, gd_downsample)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

@AbstractBuildFactory.register('model')
class SwinTransformer3DWithSBP(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 graddrop_config={"gd_downsample": 1, "with_gd": [[1, 1], [1, 1], [1] * 4 + [0] * 2, [0, 0]]},):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint,
                with_gd=graddrop_config["with_gd"][i_layer])
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()

        # graddrop
        self.gd_downsample = graddrop_config["gd_downsample"]  # int.
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = (
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(
                                1, nH1, S1, S1
                            ),
                            size=(
                                2 * self.window_size[1] - 1,
                                2 * self.window_size[2] - 1,
                            ),
                            mode="bicubic",
                        )
                    )
                    relative_position_bias_table_pretrained = (
                        relative_position_bias_table_pretrained_resized.view(
                            nH2, L2
                        ).permute(1, 0)
                    )
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1
            )

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()
    
    def _clear_memory_buffer(self):
        pass

    def init_weights(self, child_model=False, revise_keys=[(r'backbone.', r''), (r'norm2', r'mlp.norm')]):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if child_model is False:
            if isinstance(self.pretrained, str):
                self.apply(_init_weights)
                logger = get_logger("SVTAS")
                logger.info(f'load model from: {self.pretrained}')

                if self.pretrained2d:
                    # Inflate 2D model into 3D model.
                    self.inflate_weights(logger)
                else:
                    # Directly load 3D model.
                    load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=[(r'backbone.', r''), (r'norm2', r'mlp.norm')])
            elif self.pretrained is None:
                self.apply(_init_weights)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            self.apply(_init_weights)
    
    def generate_grad_mask(self, x, gd_downsample):
        """generate mask for grad drop.

        Args:
            x (torch.Tensor): input.
            gd_downsample (int): graddrop downsample rate.

        Returns: torch.Tensor[B,D]. The mask for graddrop.

        """
        B, C, D, H, W = x.shape
        # grad_mask = np.indices([B, D]).sum(0) % gd_downsample == 0
        # grad_mask = torch.from_numpy(grad_mask).to(x.device)
        grad_mask = torch.zeros([B, D], dtype=torch.bool).to(x.device)
        grad_mask[:, ::gd_downsample] = True
        grad_mask = torch.roll(grad_mask, gd_downsample // 2, -1)  # roll to the center.
        # print(grad_mask)
        return grad_mask

    def forward(self, x, masks):
        """Forward function."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            grad_mask = self.generate_grad_mask(x, self.gd_downsample)
            x = layer(x.contiguous(), grad_mask, self.gd_downsample)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        
        x = x * F.adaptive_max_pool3d(masks, output_size=[x.shape[2], 1, 1])
        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3DWithSBP, self).train(mode)
        self._freeze_stages()

