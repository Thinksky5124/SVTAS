'''
Author       : Thyssen Wen
Date         : 2022-10-27 17:12:13
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 18:17:41
Description  : Stream Base Dataset
FilePath     : /SVTAS/loader/dataset/stream_base_dataset/__init__.py
'''
from .feature_stream_segmentation_dataset import \
    FeatureStreamSegmentationDataset
from .feature_video_prediction_dataset import FeatureVideoPredictionDataset
from .raw_frame_stream_segmentation_dataset import \
    RawFrameStreamSegmentationDataset
from .rgb_flow_frame_stream_segmentation_dataset import \
    RGBFlowFrameStreamSegmentationDataset
from .video_cam_raw_frame_stream_dataset import RawFrameStreamCAMDataset

__all__ = [
    "FeatureStreamSegmentationDataset", "FeatureVideoPredictionDataset", "RawFrameStreamSegmentationDataset",
    "RGBFlowFrameStreamSegmentationDataset", "RawFrameStreamCAMDataset"
]