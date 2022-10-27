'''
Author       : Thyssen Wen
Date         : 2022-05-18 21:30:31
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-26 12:57:15
Description  : Image Cliassification Field model zoom
FilePath     : /SVTAS/model/backbones/image/__init__.py
'''
from .mobilenet_v2 import MobileNetV2
from .mobilevit import MobileViT
from .resnet import ResNet
from .vit import ViT
from .swin_v2_transformer import SwinTransformerV2
from .vit_for_small_dataset import SLViT
from .sample_vit import SimpleViT
from .clip import CLIP

__all__ = [
    "MobileNetV2", "MobileViT", "ResNet", "ViT", "SwinTransformerV2",
    "SLViT", "SimpleViT", "CLIP"
]