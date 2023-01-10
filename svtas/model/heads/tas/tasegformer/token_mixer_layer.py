'''
Author       : Thyssen Wen
Date         : 2022-12-30 16:11:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-01-09 16:54:50
Description  : file content
FilePath     : /SVTAS/svtas/model/heads/tas/tasegformer/token_mixer_layer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from .....utils.logger import get_logger, tensorboard_log_feature_image

class PoolFormerMixTokenLayer(nn.Module):
    """
    Implementation of pooling for PoolFormer
    implement by pytorch (ref:https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py),
    from paper <MetaFormer is Actually What You Need for Vision> :https://arxiv.org/pdf/2111.11418.pdf

    --pool_size: pooling size
    """
    def __init__(self, pool_size=3, stride=1, mode='encoder'):
        super().__init__()
        self.mode = mode
        assert mode in ['encoder', 'decoder'], f"mode {mode} do not support!"
        if pool_size % 2 == 0:
            pool_size += 1
        # if self.mode == 'encoder':
        self.pool = nn.AvgPool1d(
            pool_size, stride=stride, padding=pool_size//2, count_include_pad=False)
        # elif self.mode == 'decoder':
        #     self.pool = nn.AvgPool1d(
        #         pool_size, stride=stride, padding=pool_size//2, count_include_pad=False)

    def forward(self, x, masks):
        x = self.pool(x) - x
        return x * masks[:, 0:1, :]

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, T]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, mode='encoder', pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, drop=0., stride=1,
                 use_layer_scale=True, layer_scale_init_value=1e-3):

        super().__init__()

        self.norm1 = nn.InstanceNorm1d(dim)
        self.token_mixer = PoolFormerMixTokenLayer(pool_size=pool_size, stride=stride, mode=mode)
        self.norm2 = nn.InstanceNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, masks):
        if self.use_layer_scale:
            writer = get_logger(tensorboard=True)
            tensorboard_log_feature_image(writer, x[0], epoch=0, tag="raw_x")

            out = self.token_mixer(x, masks)
            tensorboard_log_feature_image(writer, out[0], epoch=0, tag="token_mixer")

            sclae_x = self.layer_scale_1.unsqueeze(-1) * out
            tensorboard_log_feature_image(writer, sclae_x[0], epoch=0, tag="token_sclae")

            norm_x = self.norm1(sclae_x)
            tensorboard_log_feature_image(writer, norm_x[0], epoch=0, tag="token_norm")

            x = x + norm_x
            tensorboard_log_feature_image(writer, x[0], epoch=0, tag="token_res")

            norm_x = self.norm2(x)
            tensorboard_log_feature_image(writer, norm_x[0], epoch=0, tag="ffn_norm")

            ff_x = self.ffn(norm_x)
            tensorboard_log_feature_image(writer, ff_x[0], epoch=0, tag="ffn")

            out = self.layer_scale_2.unsqueeze(-1) * ff_x
            tensorboard_log_feature_image(writer, out[0], epoch=0, tag="ffn_scale")

            x = x + out
            tensorboard_log_feature_image(writer, x[0], epoch=0, tag="ffn_res")

        else:
            x = x + self.norm1(self.token_mixer(x, masks))
            x = x + self.ffn(self.norm2(x))
        return x