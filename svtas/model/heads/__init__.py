'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 13:58:13
Description: Head init
FilePath     : /SVTAS/svtas/model/heads/__init__.py
'''
from .segmentation import (MultiStageModel, ETESVSHead, ASFormer, LSTMSegmentationHead,
                        MultiStageModel, SingleStageModel, TCN3DHead)
from .recognition import (TSMHead, I3DHead, MoViNetHead, TimeSformerHead, FCHead)
from .online_action_detection import (OadTRHead, LSTR)
from .feature_extractor import FeatureExtractHead
from .joint import TransudcerJointNet, TransegerFCJointNet
from .text_pred import TextPredFCHead
from .automatic_speech_recognition import Conformer
from .align_heads import (InterploteAlignHead)

__all__ = [
    'MultiStageModel', 'ETESVSHead', 'ASFormer', 'MultiStageModel',
    'SingleStageModel', 'TCN3DHead', 'TSMHead', 'I3DHead', 'MoViNetHead',
    'TimeSformerHead', 'LSTMSegmentationHead', 'FCHead', 'OadTRHead',
    'FeatureExtractHead', 'TransudcerJointNet', 'TransegerFCJointNet',
    'TextPredFCHead', 'Conformer', 'InterploteAlignHead', "LSTR"
]