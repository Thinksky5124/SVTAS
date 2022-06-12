'''
Author       : Thyssen Wen
Date         : 2022-05-18 21:30:31
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-12 21:02:25
Description  : Image Cliassification Field model zoom
FilePath     : /ETESVS/model/backbones/image/__init__.py
'''
from .mobilenet_v2 import MobileNetV2
from .mobilevit import MobileViT
from .resnet import ResNet
from .vit import ViT
from .swin_v2_transformer import SwinTransformerV2

__all__ = [
    "MobileNetV2", "MobileViT", "ResNet", "ViT", "SwinTransformerV2"
]