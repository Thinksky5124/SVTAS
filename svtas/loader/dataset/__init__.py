'''
Author       : Thyssen Wen
Date         : 2022-05-18 14:55:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-20 10:49:25
Description  : datatset class
FilePath     : /SVTAS/svtas/loader/dataset/__init__.py
'''
from .item_base_dataset import (FeatureSegmentationDataset,
                                RawFrameSegmentationDataset,
                                RawFrameClipSegmentationDataset,
                                CAMFeatureSegmentationDataset)
from .stream_base_dataset import (FeatureStreamSegmentationDataset,
                                  FeatureVideoPredictionDataset,
                                  RawFrameStreamCAMDataset,
                                  RawFrameStreamSegmentationDataset,
                                  RGBFlowFrameStreamSegmentationDataset,
                                  CompressedVideoStreamSegmentationDataset,
                                  RGBMVsResFrameStreamSegmentationDataset,
                                  CAMFeatureStreamSegmentationDataset)

__all__ = [
    'RawFrameStreamSegmentationDataset', 'FeatureStreamSegmentationDataset',
    'RGBFlowFrameStreamSegmentationDataset',
    'FeatureVideoPredictionDataset', 'FeatureSegmentationDataset',
    'RawFrameSegmentationDataset', 'RawFrameStreamCAMDataset',
    'CompressedVideoStreamSegmentationDataset',
    'RGBMVsResFrameStreamSegmentationDataset',
    'RawFrameClipSegmentationDataset',
    'CAMFeatureSegmentationDataset',
    'CAMFeatureStreamSegmentationDataset'
]