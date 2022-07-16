'''
Author: Thyssen Wen
Date: 2022-04-14 16:04:39
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-16 09:59:09
Description: Mobilenet V2 TSM model ref:https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/backbones/mobilenet_v2_tsm.py
FilePath     : /ETESVS/model/backbones/video/mobilenet_v2_tsm.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from ..image.mobilenet_v2 import InvertedResidual, MobileNetV2
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from utils.logger import get_logger
from .resnet_tsm import TemporalShift

from ...builder import BACKBONES

@BACKBONES.register()
class MobileNetV2TSM(MobileNetV2):
    """MobileNetV2 backbone for TSM.

    Args:
        num_segments (int): Number of frame segments. Default: 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Default: True.
        shift_div (int): Number of div for shift. Default: 8.
        **kwargs (keyword arguments, optional): Arguments for MobilNetV2.
    """

    def __init__(self, clip_seg_num=8, is_shift=True, shift_div=8, modality="RGB", **kwargs):
        super().__init__(**kwargs)
        self.num_segments = clip_seg_num
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.modality = modality

    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        for m in self.modules():
            if isinstance(m, InvertedResidual) and \
                    len(m.conv) == 3 and m.use_res_connect:
                m.conv[0] = TemporalShift(
                    m.conv[0],
                    num_segments=self.num_segments,
                    shift_div=self.shift_div,
                )
    
    def _clear_memory_buffer(self):
        pass

    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.is_shift:
            self.make_temporal_shift()
            
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("SVTAS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)
            elif self.pretrained is None:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                        constant_init(m, 1)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                        constant_init(m, 1)