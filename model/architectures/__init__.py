'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-04 15:09:27
Description: file content
FilePath     : /ETESVS/model/architectures/__init__.py
'''
from .segmentation.stream_segmentation2d_with_neck import StreamSegmentation2DWithNeck
from .segmentation.feature_segmentation import FeatureSegmentation
from .recognition.recognition2d import Recognition2D
from .recognition.recognition3d import Recognition3D
from .segmentation.stream_segmentation2d import StreamSegmentation2D
from .segmentation.stream_segmentation3d import StreamSegmentation3D
from .segmentation.multi_modality_stream_segmentation import MulModStreamSegmentation
from .optical_flow.optical_flow_estimator import OpticalFlowEstimation
from .segmentation.transeger import Transeger
from .general.encoder_2_decoder import Encoder2Decoder

__all__ = [
    'StreamSegmentation2DWithNeck', 'FeatureSegmentation',
    'Recognition2D', 'Recognition3D',
    'StreamSegmentation2D', 'StreamSegmentation3D',
    'MulModStreamSegmentation',
    'OpticalFlowEstimation',
    'Transeger', 'Encoder2Decoder'
]