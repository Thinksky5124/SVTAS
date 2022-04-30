'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:53
LastEditors: Thyssen Wen
LastEditTime: 2022-04-29 11:04:06
Description: file content
FilePath: /ETESVS/model/losses/__init__.py
'''
from .etesvs_loss import ETESVSLoss
from .mstcn_loss import MSTCNLoss
from .tsm_loss import TSMLoss

__all__ = [
    'ETESVSLoss', 'MSTCNLoss', 'TSMLoss'
]