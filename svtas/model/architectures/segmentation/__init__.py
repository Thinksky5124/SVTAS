'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:10:45
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 15:29:03
Description  : Segmentation Framweork
FilePath     : /SVTAS/svtas/model/architectures/segmentation/__init__.py
'''
from .action_clip_segmentation import ActionCLIPSegmentation
from .feature_segmentation import FeatureSegmentation
from .multi_modality_stream_segmentation import MultiModalityStreamSegmentation
from .stream_action_clip_segmentation import StreamSegmentationActionCLIPWithBackbone
from .stream_video_segmentation import StreamVideoSegmentation
from .video_segmentation import VideoSegmentation
from .transeger import Transeger

__all__ = [
    'ActionCLIPSegmentation', 'FeatureSegmentation', 'MultiModalityStreamSegmentation',
    'StreamSegmentationActionCLIPWithBackbone', 'StreamVideoSegmentation',
    'VideoSegmentation', 'Transeger'
]