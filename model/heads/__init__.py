'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-21 13:50:40
Description: Head init
FilePath     : /ETESVS/model/heads/__init__.py
'''
from .segmentation.mstcn import MultiStageModel
from .segmentation.etesvs_head import ETESVSHead
from .segmentation.asformer import ASFormer
from .segmentation.mstcn import MultiStageModel, SingleStageModel
from .segmentation.tcn_3d_head import TCN3DHead
from .recognition.tsm_head import TSMHead
from .recognition.i3d_head import I3DHead
from .recognition.movinet_head import MoViNetHead
from .recognition.timesformer_head import TimeSformerHead
from .segmentation.lstm_head import LSTMSegmentationHead
from .recognition.fc_head import FCHead
from .online_action_detection.oadtr import OadTRHead
from .feature_extractor.feature_extract_head import FeatureExtractHead
from .segmentation.transducer_joint_head import TransudcerJointNet

__all__ = [
    'MultiStageModel', 'ETESVSHead', 'ASFormer', 'MultiStageModel',
    'SingleStageModel', 'TCN3DHead', 'TSMHead', 'I3DHead', 'MoViNetHead',
    'TimeSformerHead', 'LSTMSegmentationHead', 'FCHead', 'OadTRHead',
    'FeatureExtractHead', 'TransudcerJointNet'
]