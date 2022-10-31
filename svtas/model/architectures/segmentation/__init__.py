'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:10:45
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:10:53
Description  : Segmentation Framweork
FilePath     : /SVTAS/svtas/model/architectures/segmentation/__init__.py
'''
from .video import (Segmentation3D, Segmentation2D)
from .stream_video import (StreamSegmentation2DWithNeck, StreamSegmentation2DWithBackbone, StreamSegmentation3DWithBackbone,
                           MulModStreamSegmentation, Transeger, StreamSegmentation2D, StreamSegmentation3D)
from .feature import (FeatureSegmentation)

__all__ = [
    'StreamSegmentation2DWithNeck', 'FeatureSegmentation',
    'StreamSegmentation2DWithBackbone', 'StreamSegmentation3DWithBackbone',
    'MulModStreamSegmentation',
    'Transeger',
    'StreamSegmentation2D',
    'StreamSegmentation3D',
    'Segmentation3D', 'Segmentation2D'
]