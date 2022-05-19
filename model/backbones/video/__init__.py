'''
Author       : Thyssen Wen
Date         : 2022-05-18 21:30:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 21:41:56
Description  : Video Filed model zoom
FilePath     : /ETESVS/model/backbones/video/__init__.py
'''
from .i3d import I3D
from .mobilenet_v2_tsm import MobileNetV2TSM
from .movinet import MoViNet
from .resnet2plus1d import ResNet2Plus1d
from .predrnn_v2 import PredRNNV2
from .resnet_3d import ResNet3d
from .timesfromer import TimeSformer
from .resnet_tsm import ResNetTSM

__all__ = [
    "I3D", "MobileNetV2TSM", "MoViNet", "ResNet2Plus1d", "PredRNNV2", "ResNet3d",
    "TimeSformer", "ResNetTSM"
]