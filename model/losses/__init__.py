'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-11 10:43:11
Description: file content
FilePath     : /ETESVS/model/losses/__init__.py
'''
from .etesvs_loss import ETESVSLoss
from .segmentation_loss import SegmentationLoss
from .recognition_segmentation_loss import RecognitionSegmentationLoss
from .recognition_segmentation_loss import SoftLabelRocgnitionLoss
from .steam_segmentation_loss import StreamSegmentationLoss

__all__ = [
    'ETESVSLoss', 'SegmentationLoss', 'RecognitionSegmentationLoss',
    'StreamSegmentationLoss', 'SoftLabelRocgnitionLoss'
]