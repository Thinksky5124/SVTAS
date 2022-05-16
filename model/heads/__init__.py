'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:46
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-16 21:07:57
Description: Head init
FilePath     : /ETESVS/model/heads/__init__.py
'''
from .mstcn import MultiStageModel
from .etesvs_head import ETESVSHead
from .asformer import ASFormer
from .mstcn import MultiStageModel, SingleStageModel
from .tcn_3d_head import TCN3DHead
from .tsm_head import TSMHead
from .i3d_head import I3DHead
from .movinet_head import MoViNetHead
from .timesformer_head import TimeSformerHead
from .lstm_head import LSTMSegmentationHead
from .fc_head import FCHead

__all__ = [
    'MultiStageModel', 'ETESVSHead', 'ASFormer', 'MultiStageModel',
    'SingleStageModel', 'TCN3DHead', 'TSMHead', 'I3DHead', 'MoViNetHead',
    'TimeSformerHead', 'LSTMSegmentationHead', 'FCHead'
]