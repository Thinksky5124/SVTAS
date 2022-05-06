'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 15:49:09
Description: file content
FilePath     : /ETESVS/model/architectures/__init__.py
'''
from .stream_segmentation_hold import StreamSegmentationWithNeck
from .feature_segmentation import FeatureSegmentation
from .recognition2d import Recognition2D
from .recognition3d import Recognition3D
from .stream_segmentation import StreamSegmentation
from .multi_modality_stream_segmentation import MulModStreamSegmentation
from .optical_flow_estimator import OpticalFlowEstimation

__all__ = [
    'StreamSegmentationWithNeck', 'FeatureSegmentation',
    'Recognition2D', 'Recognition3D',
    'StreamSegmentation', 'MulModStreamSegmentation',
    'OpticalFlowEstimation'
]