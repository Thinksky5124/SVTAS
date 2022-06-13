'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:10:45
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-13 16:24:19
Description  : Segmentation Framweork
FilePath     : /ETESVS/model/architectures/segmentation/__init__.py
'''
from .stream_segmentation2d_with_neck import StreamSegmentation2DWithNeck
from .feature_segmentation import FeatureSegmentation
from .stream_segmentation2d import StreamSegmentation2D
from .stream_segmentation3d import StreamSegmentation3D
from .multi_modality_stream_segmentation import MulModStreamSegmentation
from .transeger import Transeger
from .segmentation_clip import SegmentationCLIP
from .stream_segmentation2d_without_backbone import StreamSegmentation2DWithoutBackbone
from .stream_segmentation3d_without_backbone import StreamSegmentation3DWithoutBackbone

__all__ = [
    'StreamSegmentation2DWithNeck', 'FeatureSegmentation',
    'StreamSegmentation2D', 'StreamSegmentation3D',
    'MulModStreamSegmentation',
    'Transeger', 'SegmentationCLIP',
    'StreamSegmentation2DWithoutBackbone',
    'StreamSegmentation3DWithoutBackbone'
]