'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-28 08:48:48
Description  : Segmentation Head Modules
FilePath     : /SVTAS/svtas/model/heads/tas/__init__.py
'''
from .asformer import ASFormer
from .mstcn import MultiStageModel, SingleStageModel
from .tcn_3d_head import TCN3DHead
from .lstm_head import LSTMSegmentationHead
from .memory_tcn import MemoryTCNHead
from .linear_transformer import LinformerHead
from .tasegformer import TASegFormer
from .asrf import ActionSegmentRefinementFramework
from .c2f_tcn import C2F_TCN
from .transformer_xl import TransformerXL
from .block_recurrent_transformer import BRTSegmentationHead, ASRFWithBRT, BRTClassificationHead

__all__ = [
    'MultiStageModel', 'ASFormer', 'SingleStageModel',
    'TCN3DHead', 'LSTMSegmentationHead', 'MemoryTCNHead', 'LinformerHead',
    'TASegFormer', 'ActionSegmentRefinementFramework', 'C2F_TCN',
    'TransformerXL', 'BRTSegmentationHead', 'ASRFWithBRT',
    'BRTClassificationHead'
]