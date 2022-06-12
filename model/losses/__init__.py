'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-11 11:48:13
Description: file content
FilePath     : /ETESVS/model/losses/__init__.py
'''
from .etesvs_loss import ETESVSLoss
from .segmentation_loss import SegmentationLoss
from .recognition_segmentation_loss import RecognitionSegmentationLoss
from .recognition_segmentation_loss import SoftLabelRocgnitionLoss
from .steam_segmentation_loss import StreamSegmentationLoss
from .video_prediction_loss import VideoPredictionLoss
from .transeger_loss import TransegerLoss
from .segmentation_clip_loss import SgementationCLIPLoss, CLIPLoss

__all__ = [
    'ETESVSLoss', 'SegmentationLoss', 'RecognitionSegmentationLoss',
    'StreamSegmentationLoss', 'SoftLabelRocgnitionLoss', 'VideoPredictionLoss',
    'TransegerLoss', 'SgementationCLIPLoss', 'CLIPLoss'
]