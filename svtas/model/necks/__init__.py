'''
Author       : Thyssen Wen
Date         : 2022-10-17 13:15:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-27 17:25:29
Description  : neck registration
FilePath     : /SVTAS/svtas/model/necks/__init__.py
'''
from .etesvs_neck import ETESVSNeck
from .memory_layer import ConvLSTMResidualLayer
from .st_lstm_3d_neck import LSTMST3DNeck
from .pool_neck import PoolNeck
from .action_clip_fusion_model import ActionCLIPFusionNeck
from .bridge_fusion_earlyhyp import BridgePromptFusionEarlyhyp
from .multimodality_fusion_neck import MultiModalityFusionNeck
from .ipb_fusion_neck import IPBFusionNeck
from .unsample_decoder_neck import UnsampleDecoderNeck
from .task_fuion_neck import TaskFusionPoolNeck

__all__ = [
    'ETESVSNeck', 'ConvLSTMResidualLayer', 'LSTMST3DNeck', 'PoolNeck',
    'ActionCLIPFusionNeck', 'BridgePromptFusionEarlyhyp',
    'MultiModalityFusionNeck', 'IPBFusionNeck', 'UnsampleDecoderNeck',
    'TaskFusionPoolNeck'
]