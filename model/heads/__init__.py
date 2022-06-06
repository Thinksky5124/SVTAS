'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-06 20:26:50
Description: Head init
FilePath     : /ETESVS/model/heads/__init__.py
'''
from .segmentation import (MultiStageModel, ETESVSHead, ASFormer, LSTMSegmentationHead,
                        MultiStageModel, SingleStageModel, TCN3DHead)
from .recognition import (TSMHead, I3DHead, MoViNetHead, TimeSformerHead, FCHead)
from .online_action_detection import OadTRHead
from .feature_extractor import FeatureExtractHead
from .joint import TransudcerJointNet, TransegerJointNet
from .text_pred import TextPredFCHead

__all__ = [
    'MultiStageModel', 'ETESVSHead', 'ASFormer', 'MultiStageModel',
    'SingleStageModel', 'TCN3DHead', 'TSMHead', 'I3DHead', 'MoViNetHead',
    'TimeSformerHead', 'LSTMSegmentationHead', 'FCHead', 'OadTRHead',
    'FeatureExtractHead', 'TransudcerJointNet', 'TransegerJointNet',
    'TextPredFCHead'
]