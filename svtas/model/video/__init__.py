'''
Author       : Thyssen Wen
Date         : 2022-05-18 21:30:25
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 19:33:49
Description  : Video Filed model zoom
FilePath     : /SVTAS/svtas/model/backbones/video/__init__.py
'''
from .i3d import I3D
from .mobilenet_v2_tsm import MobileNetV2TSM
from .movinet import MoViNet
from .resnet2plus1d import ResNet2Plus1d
from .predrnn_v2 import PredRNNV2
from .resnet_3d import ResNet3d
from .resnet_tsm import ResNetTSM
from .swin_transformer_3d import SwinTransformer3D
from .vit_3d import ViT3D
from .sample_vit_3d import SampleViT3D
from .mvit import MViT
from .x3d import X3D
from .swin_transformer_3d_sbp import SwinTransformer3DWithSBP
from .resnet3d_slowfast import ResNet3dSlowFast
from .resnet3d_slowonly import ResNet3dSlowOnly

__all__ = [
    "I3D", "MobileNetV2TSM", "MoViNet", "ResNet2Plus1d", "PredRNNV2", "ResNet3d",
    "TimeSformer", "ResNetTSM", "SwinTransformer3D", "ViT3D", "SampleViT3D",
    "MViT", "X3D", "SwinTransformer3DWithSBP", "ResNet3dSlowFast",
    "ResNet3dSlowOnly"
]