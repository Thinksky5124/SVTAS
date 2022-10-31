'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:11:02
Description: file content
FilePath     : /SVTAS/svtas/model/architectures/__init__.py
'''
from .segmentation import (StreamSegmentation2DWithNeck, FeatureSegmentation,
                        StreamSegmentation2D, StreamSegmentation3D,
                        MulModStreamSegmentation, Transeger)

from .recognition import (Recognition2D, Recognition3D, ActionCLIP)
from .optical_flow import OpticalFlowEstimation
from .general import Encoder2Decoder

__all__ = [
    'StreamSegmentation2DWithNeck', 'FeatureSegmentation',
    'Recognition2D', 'Recognition3D', 'ActionCLIP',
    'StreamSegmentation2D', 'StreamSegmentation3D',
    'MulModStreamSegmentation',
    'OpticalFlowEstimation',
    'Transeger', 'Encoder2Decoder'
]