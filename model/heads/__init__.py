'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:46
LastEditors: Thyssen Wen
LastEditTime: 2022-04-30 14:43:13
Description: file content
FilePath: /ETESVS/model/heads/__init__.py
'''
from .mstcn import MultiStageModel
from .etesvs_head import ETESVSHead
from .asformer import ASFormer
from .mstcn import MultiStageModel, SingleStageModel
from .tcn_3d_head import TCN3DHead
from .tsm_head import TSMHead
from .i3d_head import I3DHead

__all__ = [
    'MultiStageModel', 'ETESVSHead', 'ASFormer', 'MultiStageModel',
    'SingleStageModel', 'TCN3DHead', 'TSMHead', 'I3DHead'
]