'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-30 16:06:01
Description: Head init
FilePath     : /SVTAS/svtas/model/heads/__init__.py
'''
from .tas import (MultiStageModel, ASFormer, LSTMSegmentationHead,
                        MultiStageModel, SingleStageModel, TCN3DHead)
from .recognition import (TSMHead, I3DHead, MoViNetHead, TimeSformerHead, FCHead)
from .oad import (OadTRHead, LSTR)
from .fe import FeatureExtractHead
from .joint import TransudcerJointNet, TransegerFCJointNet
from .text_pred import TextPredFCHead
from .asr import Conformer
from .align_heads import (InterploteAlignHead)

__all__ = [
    'MultiStageModel', 'ASFormer', 'MultiStageModel',
    'SingleStageModel', 'TCN3DHead', 'TSMHead', 'I3DHead', 'MoViNetHead',
    'TimeSformerHead', 'LSTMSegmentationHead', 'FCHead', 'OadTRHead',
    'FeatureExtractHead', 'TransudcerJointNet', 'TransegerFCJointNet',
    'TextPredFCHead', 'Conformer', 'InterploteAlignHead', "LSTR"
]