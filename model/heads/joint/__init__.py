'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 21:20:35
Description  : Joint Head Modules
FilePath     : /ETESVS/model/heads/joint/__init__.py
'''
from .transducer_joint_head import TransudcerJointNet
from .transeger_fc_joint_head import TransegerFCJointNet
from .transeger_tcn_joint_head import TransegerTCNJointNet
from .transeger_transformer_joint_head import TransegerTransformerJointNet
from .bridge_fusion_earlyhyp import BridgePromptFusionEarlyhyp

__all__ = [
    "TransudcerJointNet", "TransegerFCJointNet", "TransegerTCNJointNet",
    "TransegerTransformerJointNet", "BridgePromptFusionEarlyhyp"
]