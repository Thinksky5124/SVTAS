'''
Author       : Thyssen Wen
Date         : 2022-12-22 10:42:37
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-22 11:51:13
Description  : file content
FilePath     : /SVTAS/svtas/model/backbones/video/swin_transformer_tsm.py
'''

import torch
from mmcv.runner import load_state_dict
from .resnet_tsm import TemporalShift
from ..image.swin_v2_transformer import SwinTransformerV2
from ....utils.logger import get_logger
from ...builder import BACKBONES

class ViTTemporalShift(TemporalShift):
    """Temporal shift module for ViT.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """
    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, P, C]
        n, p, c = x.size()

        # [N, P, C] -> [N, C, P]
        x = torch.permute(x, dims=[0, 2, 1])

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, p)

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, C, P]
        # restore the original dimension
        return out.view(n, c, p).permute([0, 2, 1])
        
@BACKBONES.register()
class SwinTransformerV2TSM(SwinTransformerV2):
    def __init__(self, clip_seg_num=8, is_shift=True, shift_div=8, **kwargs):
        super().__init__(**kwargs)
        self.num_segments = clip_seg_num
        self.is_shift = is_shift
        self.shift_div = shift_div
    
    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        for i, layer in enumerate(self.layers):
            for j, block in enumerate(layer.blocks):
                self.layers[i].blocks[j] = ViTTemporalShift(
                    block,
                    num_segments=self.num_segments,
                    shift_div=self.shift_div,
                )
    
    def _clear_memory_buffer(self):
        pass

    def init_weights(self, child_model=False, revise_keys=[(r'backbone.', r'')]):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("SVTAS")
                logger.info(f'load model from: {self.pretrained}')
                state_dict = torch.load(self.pretrained)['model']
                load_state_dict(self, state_dict, strict=False, logger=logger)
            elif self.pretrained is None:
                self.apply(self._init_weights)
                for bly in self.layers:
                    bly._init_respostnorm()
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            self.apply(self._init_weights)
            for bly in self.layers:
                bly._init_respostnorm()
        if self.is_shift:
            self.make_temporal_shift()