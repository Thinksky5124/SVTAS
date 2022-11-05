'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:10:45
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-05 20:43:02
Description  : Segmentation Framweork
FilePath     : /SVTAS/svtas/model/architectures/segmentation/__init__.py
'''
from .video import (Segmentation3D, Segmentation2D)
from .stream_video import (StreamSegmentation2DWithNeck, StreamSegmentation2DWithBackbone, StreamSegmentation3DWithBackbone,
                           MultiModalityStreamSegmentation, Transeger, StreamSegmentation2D, StreamSegmentation3D)
from .feature import (FeatureSegmentation, FeatureSegmentation3D)

__all__ = [
    'StreamSegmentation2DWithNeck', 'FeatureSegmentation',
    'StreamSegmentation2DWithBackbone', 'StreamSegmentation3DWithBackbone',
    'MultiModalityStreamSegmentation',
    'Transeger',
    'StreamSegmentation2D',
    'StreamSegmentation3D',
    'Segmentation3D', 'Segmentation2D',
    'FeatureSegmentation3D'
]