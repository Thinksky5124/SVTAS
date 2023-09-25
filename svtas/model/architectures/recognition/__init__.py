'''
Author       : Thyssen Wen
Date         : 2022-06-11 11:11:07
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 14:08:33
Description  : Recognition Framewwork
FilePath     : /SVTAS/svtas/model/architectures/recognition/__init__.py
'''
from .action_clip import ActionCLIP
from .recognition import Recognition, VideoRocognition

__all__ = [
    'Recognition', 'VideoRocognition', 'ActionCLIP'
]