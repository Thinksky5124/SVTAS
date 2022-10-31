'''
Author       : Thyssen Wen
Date         : 2022-10-17 13:15:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 10:22:55
Description  : neck registration
FilePath     : /SVTAS/svtas/model/necks/__init__.py
'''
from .etesvs_neck import ETESVSNeck
from .memory_layer import ConvLSTMResidualLayer
from .st_3d_neck import ST3DNeck
from .avg_pool_neck import AvgPoolNeck
from .action_clip_fusion_model import ActionCLIPFusionNeck
from .bridge_fusion_earlyhyp import BridgePromptFusionEarlyhyp

__all__ = [
    'ETESVSNeck', 'ConvLSTMResidualLayer', 'ST3DNeck', 'AvgPoolNeck',
    'ActionCLIPFusionNeck', 'BridgePromptFusionEarlyhyp'
]