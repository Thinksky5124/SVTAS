'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-27 15:56:38
Description: file content
FilePath     : /ETESVS/model/architectures/__init__.py
'''
from .stream_segmentation2d_with_neck import StreamSegmentation2DWithNeck
from .feature_segmentation import FeatureSegmentation
from .recognition2d import Recognition2D
from .recognition3d import Recognition3D
from .stream_segmentation2d import StreamSegmentation2D
from .stream_segmentation3d import StreamSegmentation3D
from .multi_modality_stream_segmentation import MulModStreamSegmentation
from .optical_flow_estimator import OpticalFlowEstimation
from .transeger3d import Transeger3D
from .encoder_2_decoder import Encoder2Decoder

__all__ = [
    'StreamSegmentation2DWithNeck', 'FeatureSegmentation',
    'Recognition2D', 'Recognition3D',
    'StreamSegmentation2D', 'StreamSegmentation3D',
    'MulModStreamSegmentation',
    'OpticalFlowEstimation',
    'Transeger3D', 'Encoder2Decoder'
]