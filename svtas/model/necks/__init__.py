'''
Author       : Thyssen Wen
Date         : 2022-10-17 13:15:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-19 14:43:47
Description  : neck registration
FilePath     : /SVTAS/svtas/model/necks/__init__.py
'''
from .etesvs_neck import ETESVSNeck
from .memory_layer import ConvLSTMResidualLayer
from .st_lstm_3d_neck import LSTMST3DNeck
from .avg_pool_neck import AvgPoolNeck
from .action_clip_fusion_model import ActionCLIPFusionNeck
from .bridge_fusion_earlyhyp import BridgePromptFusionEarlyhyp
from .multimodality_fusion_neck import MultiModalityFusionNeck
from .ipb_fusion_neck import IPBFusionNeck
from .unsample_decoder_neck import UnsampleDecoderNeck

__all__ = [
    'ETESVSNeck', 'ConvLSTMResidualLayer', 'LSTMST3DNeck', 'AvgPoolNeck',
    'ActionCLIPFusionNeck', 'BridgePromptFusionEarlyhyp',
    'MultiModalityFusionNeck', 'IPBFusionNeck', 'UnsampleDecoderNeck'
]