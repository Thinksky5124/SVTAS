'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-13 20:06:55
Description: Head init
FilePath     : /ETESVS/model/heads/__init__.py
'''
from .segmentation import (MultiStageModel, ETESVSHead, ASFormer, LSTMSegmentationHead,
                        MultiStageModel, SingleStageModel, TCN3DHead)
from .recognition import (TSMHead, I3DHead, MoViNetHead, TimeSformerHead, FCHead)
from .online_action_detection import OadTRHead
from .feature_extractor import FeatureExtractHead
from .joint import TransudcerJointNet, TransegerFCJointNet
from .text_pred import TextPredFCHead
from .automatic_speech_recognition import Conformer

__all__ = [
    'MultiStageModel', 'ETESVSHead', 'ASFormer', 'MultiStageModel',
    'SingleStageModel', 'TCN3DHead', 'TSMHead', 'I3DHead', 'MoViNetHead',
    'TimeSformerHead', 'LSTMSegmentationHead', 'FCHead', 'OadTRHead',
    'FeatureExtractHead', 'TransudcerJointNet', 'TransegerFCJointNet',
    'TextPredFCHead', 'Conformer'
]