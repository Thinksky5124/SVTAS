'''
Author       : Thyssen Wen
Date         : 2022-05-18 21:30:31
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-22 21:31:59
Description  : Image Cliassification Field model zoom
FilePath     : /SVTAS/model/backbones/image/__init__.py
'''
from .mobilenet_v2 import MobileNetV2
from .mobilevit import MobileViT
from .resnet import ResNet
from .vit import ViT
from .swin_v2_transformer import SwinTransformerV2
from .vit_for_small_dataset import SLViT

__all__ = [
    "MobileNetV2", "MobileViT", "ResNet", "ViT", "SwinTransformerV2",
    "SLViT"
]