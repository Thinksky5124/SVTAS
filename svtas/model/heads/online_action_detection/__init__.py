'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:29
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 13:58:05
Description  : Online Action Detection Head Modules
FilePath     : /SVTAS/svtas/model/heads/online_action_detection/__init__.py
'''
from .oadtr import OadTRHead
from .lstr import LSTR

__all__ = ["OadTRHead", "LSTR"]