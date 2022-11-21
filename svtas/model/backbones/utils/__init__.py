'''
Author       : Thyssen Wen
Date         : 2022-05-12 16:35:44
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 20:11:49
Description  : Network utils
FilePath     : /SVTAS/svtas/model/backbones/utils/__init__.py
'''
from .timesformer import (DividedSpatialAttentionWithNorm,
                          DividedTemporalAttentionWithNorm, FFNWithNorm)
from .conv2plus1d import Conv2plus1d
from .stlstm import SpatioTemporalLSTMCell
from .transducer import EncoderLayer, PositionalEncoding, get_attn_pad_mask
from .clip import SimpleTokenizer, Transformer
from .vit_tsm import TemporalShift_VIT
from .mvit import (MultiScaleAttention, attention_pool, Mlp, TwoStreamFusion, drop_path, round_width,
                   get_3d_sincos_pos_embed, calc_mvit_feature_geometry, PatchEmbed, MultiScaleBlock)
from .x3d import (get_norm, ResStage, VideoModelStem)
from .torchvision import (_log_api_usage_once, _make_ntuple, MLP, Conv2dNormActivation)
from .efficientnet import (make_divisible, SqueezeExcitation, InvertedResidual, EdgeResidual)

__all__ = [
    'DividedSpatialAttentionWithNorm', 'DividedTemporalAttentionWithNorm',
    'FFNWithNorm', 'Conv2plus1d', 'SpatioTemporalLSTMCell',
    'EncoderLayer', 'PositionalEncoding', 'get_attn_pad_mask',
    'SimpleTokenizer', 'Transformer', "TemporalShift_VIT",
    "MultiScaleAttention", "attention_pool", "Mlp", "TwoStreamFusion", 
    "drop_path", "round_width", "get_3d_sincos_pos_embed", "calc_mvit_feature_geometry",
    "PatchEmbed", "MultiScaleBlock", "get_norm", "VideoModelStem", "ResStage",
    "_log_api_usage_once", "_make_ntuple", "MLP", "Conv2dNormActivation",
    "make_divisible", "SqueezeExcitation", "InvertedResidual",
    "EdgeResidual"
]