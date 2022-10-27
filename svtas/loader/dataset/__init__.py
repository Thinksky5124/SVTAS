'''
Author       : Thyssen Wen
Date         : 2022-05-18 14:55:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 18:18:41
Description  : datatset class
FilePath     : /SVTAS/loader/dataset/__init__.py
'''
from .item_base_dataset import (FeatureSegmentationDataset,
                                RawFrameSegmentationDataset)
from .stream_base_dataset import (FeatureStreamSegmentationDataset,
                                  FeatureVideoPredictionDataset,
                                  RawFrameStreamCAMDataset,
                                  RawFrameStreamSegmentationDataset,
                                  RGBFlowFrameStreamSegmentationDataset)

__all__ = [
    'RawFrameStreamSegmentationDataset', 'FeatureStreamSegmentationDataset',
    'RGBFlowFrameStreamSegmentationDataset',
    'FeatureVideoPredictionDataset', 'FeatureSegmentationDataset',
    'RawFrameSegmentationDataset', 'RawFrameStreamCAMDataset'
]