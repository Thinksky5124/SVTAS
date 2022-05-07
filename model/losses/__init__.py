'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 20:23:53
Description: file content
FilePath     : /ETESVS/model/losses/__init__.py
'''
from .etesvs_loss import ETESVSLoss
from .segmentation_loss import MSTCNLoss
from .recognition_segmentation_loss import SegmentationLoss
from .steam_segmentation_loss import StreamSegmentationLoss

__all__ = [
    'ETESVSLoss', 'MSTCNLoss', 'SegmentationLoss', 'StreamSegmentationLoss'
]