'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-06 20:22:55
Description  : Recognition Head
FilePath     : /ETESVS/model/heads/recognition/__init__.py
'''
from .tsm_head import TSMHead
from .i3d_head import I3DHead
from .movinet_head import MoViNetHead
from .timesformer_head import TimeSformerHead
from .fc_head import FCHead

__all__ = [
    'TSMHead', 'I3DHead', 'MoViNetHead', 'TimeSformerHead', 'FCHead'
]