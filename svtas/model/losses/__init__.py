'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 19:38:21
Description: file content
FilePath     : /SVTAS/svtas/model/losses/__init__.py
'''
from .etesvs_loss import ETESVSLoss
from .segmentation_loss import SegmentationLoss, ActionCLIPSegmentationLoss
from .recognition_segmentation_loss import RecognitionSegmentationLoss
from .recognition_segmentation_loss import SoftLabelRocgnitionLoss
from .steam_segmentation_loss import StreamSegmentationLoss
from .video_prediction_loss import VideoPredictionLoss
from .segmentation_clip_loss import SgementationCLIPLoss, CLIPLoss
from .bridge_prompt_clip_loss import BridgePromptCLIPLoss, BridgePromptCLIPSegmentationLoss

__all__ = [
    'ETESVSLoss', 'SegmentationLoss', 'RecognitionSegmentationLoss',
    'StreamSegmentationLoss', 'SoftLabelRocgnitionLoss', 'VideoPredictionLoss',
    'SgementationCLIPLoss', 'CLIPLoss', 'BridgePromptCLIPLoss',
    'BridgePromptCLIPSegmentationLoss', 'ActionCLIPSegmentationLoss'
]