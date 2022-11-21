'''
Author       : Thyssen Wen
Date         : 2022-05-18 21:30:31
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 18:54:45
Description  : Image Cliassification Field model zoom
FilePath     : /SVTAS/svtas/model/backbones/image/__init__.py
'''
from .mobilenet_v2 import MobileNetV2
from .mobilevit import MobileViT
from .resnet import ResNet
from .swin_v2_transformer import SwinTransformerV2
from .vit_for_small_dataset import SLViT
from .sample_vit import SimpleViT
from .clip import CLIP
from .efficientformer import EfficientFormer
from .efficientnet import EfficientNet
from .mobilenet_v3 import MobileNetV3
from .vision_transformer import VisionTransformer

__all__ = [
    "MobileNetV2", "MobileViT", "ResNet", "VisionTransformer", "SwinTransformerV2",
    "SLViT", "SimpleViT", "CLIP", "EfficientFormer", "EfficientNet",
    "MobileNetV3"
]