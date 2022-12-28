'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:16
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-27 21:04:50
Description  : Segmentation Head Modules
FilePath     : /SVTAS/svtas/model/heads/segmentation/__init__.py
'''
from .etesvs_head import ETESVSHead
from .asformer import ASFormer
from .mstcn import MultiStageModel, SingleStageModel
from .tcn_3d_head import TCN3DHead
from .lstm_head import LSTMSegmentationHead
from .memory_tcn import MemoryTCNHead
from .linear_transformer import LinformerHead
from .segformer import SegFormer

__all__ = [
    'MultiStageModel', 'ETESVSHead', 'ASFormer', 'SingleStageModel',
    'TCN3DHead', 'LSTMSegmentationHead', 'MemoryTCNHead', 'LinformerHead',
    'SegFormer'
]