'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-06 20:20:25
Description  : Joint Head Modules
FilePath     : /ETESVS/model/heads/joint/__init__.py
'''
from .transducer_joint_head import TransudcerJointNet
from .transeger_joint_head import TransegerJointNet

__all__ = ["TransudcerJointNet", "TransegerJointNet"]