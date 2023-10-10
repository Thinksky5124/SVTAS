'''
Author       : Thyssen Wen
Date         : 2022-10-27 17:12:13
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-20 10:49:09
Description  : Stream Base Dataset
FilePath     : /SVTAS/svtas/loader/dataset/stream_base_dataset/__init__.py
'''
from .feature_stream_segmentation_dataset import \
    FeatureStreamSegmentationDataset
from .feature_video_prediction_dataset import FeatureVideoPredictionDataset
from .raw_frame_stream_segmentation_dataset import \
    RawFrameStreamSegmentationDataset
from .rgb_flow_frame_stream_segmentation_dataset import \
    RGBFlowFrameStreamSegmentationDataset
from .video_cam_raw_frame_stream_dataset import RawFrameStreamCAMDataset
from .compressed_video_stream_segmentation_dataset import CompressedVideoStreamSegmentationDataset
from .rgb_mvs_res_stream_segmentation_dataset import RGBMVsResFrameStreamSegmentationDataset
from .cam_feature_stream_segmentation_dataset import CAMFeatureStreamSegmentationDataset
from .feature_dynamic_stream_segmentation_dataset import FeatureDynamicStreamSegmentationDataset
from .raw_frame_dynamic_stream_segmentation_dataset import RawFrameDynamicStreamSegmentationDataset

__all__ = [
    "FeatureStreamSegmentationDataset", "FeatureVideoPredictionDataset", "RawFrameStreamSegmentationDataset",
    "RGBFlowFrameStreamSegmentationDataset", "RawFrameStreamCAMDataset", "CompressedVideoStreamSegmentationDataset",
    "RGBMVsResFrameStreamSegmentationDataset", "CAMFeatureStreamSegmentationDataset",
    "FeatureDynamicStreamSegmentationDataset", "RawFrameDynamicStreamSegmentationDataset"
]