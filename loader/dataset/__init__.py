'''
Author       : Thyssen Wen
Date         : 2022-05-18 14:55:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 16:16:10
Description  : datatset class
FilePath     : /ETESVS/loader/dataset/__init__.py
'''
from .raw_frame_segmentation_dataset import RawFrameSegmentationDataset
from .feature_segmentation_dataset import FeatureSegmentationDataset
from .rgb_flow_frame_segmentation_dataset import RGBFlowFrameSegmentationDataset
from .feature_video_prediction_dataset import FeatureVideoPredictionDataset

__all__ = [
    'RawFrameSegmentationDataset', 'FeatureSegmentationDataset',
    'RGBFlowFrameSegmentationDataset',
    'FeatureVideoPredictionDataset'
]