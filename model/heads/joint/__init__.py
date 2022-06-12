'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-12 17:12:54
Description  : Joint Head Modules
FilePath     : /ETESVS/model/heads/joint/__init__.py
'''
from .transducer_joint_head import TransudcerJointNet
from .transeger_fc_joint_head import TransegerFCJointNet
from .transeger_tcn_joint_head import TransegerTCNJointNet
from .transeger_transformer_joint_head import TransegerTransformerJointNet

__all__ = [
    "TransudcerJointNet", "TransegerFCJointNet", "TransegerTCNJointNet",
    "TransegerTransformerJointNet"
]