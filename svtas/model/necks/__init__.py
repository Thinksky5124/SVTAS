'''
Author       : Thyssen Wen
Date         : 2022-10-17 13:15:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-26 13:15:34
Description  : neck registration
FilePath     : /SVTAS/model/necks/__init__.py
'''
from .etesvs_neck import ETESVSNeck
from .memory_layer import ConvLSTMResidualLayer
from .st_3d_neck import ST3DNeck
from .avg_pool_neck import AvgPoolNeck
from .action_clip_fusion_model import ActionCLIPFusionNeck

__all__ = [
    'ETESVSNeck', 'ConvLSTMResidualLayer', 'ST3DNeck', 'AvgPoolNeck',
    'ActionCLIPFusionNeck'
]