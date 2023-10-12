'''
Author: Thyssen Wen
Date: 2022-03-25 19:37:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 19:35:42
Description: TSM ref: https://github.com/open-mmlab/mmaction2
FilePath     : /SVTAS/svtas/model/backbones/video/resnet_tsm.py
'''
import abc
import torch
import torch.nn as nn
from typing import Optional, Dict

from torch.nn.modules.utils import _ntuple
from svtas.model_pipline.torch_utils import (constant_init, kaiming_init,
                                             ConvModule, normal_init, load_checkpoint)
from svtas.utils.logger import get_logger

from ..image.resnet import ResNet

from svtas.utils import AbstractBuildFactory


class _NonLocalNd(nn.Module, metaclass=abc.ABCMeta):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2,
                 use_scale: bool = True,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 mode: str = 'embedded_gaussian',
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode

        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)  # type: ignore
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.mode != 'gaussian':
            self.theta = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)
            self.phi = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)

        if self.mode == 'concatenation':
            self.concat_project = ConvModule(
                self.inter_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ReLU'))

        self.init_weights(**kwargs)

    def init_weights(self, std: float = 0.01, zeros_init: bool = True) -> None:
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                normal_init(m.conv, std=std)
        else:
            normal_init(self.g.conv, std=std)
        if zeros_init:
            if self.conv_out.norm_cfg is None:
                constant_init(self.conv_out.conv, 0)
            else:
                constant_init(self.conv_out.norm, 0)
        else:
            if self.conv_out.norm_cfg is None:
                normal_init(self.conv_out.conv, std=std)
            else:
                normal_init(self.conv_out.norm, std=std)

    def gaussian(self, theta_x: torch.Tensor,
                 phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x: torch.Tensor,
                          phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x: torch.Tensor,
                    phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x: torch.Tensor,
                      phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])

        output = x + self.conv_out(y)

        return output
    
class NonLocal3d(_NonLocalNd):
    """3D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv3d').
    """

    def __init__(self,
                 in_channels: int,
                 sub_sample: bool = False,
                 conv_cfg: Dict = dict(type='Conv3d'),
                 **kwargs):
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)
        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer

class NL3DWrapper(nn.Module):
    """3D Non-local wrapper for ResNet50.

    Wrap ResNet layers with 3D NonLocal modules.

    Args:
        block (nn.Module): Residual blocks to be built.
        num_segments (int): Number of frame segments.
        non_local_cfg (dict): Config for non-local layers. Default: ``dict()``.
    """

    def __init__(self, block, num_segments, non_local_cfg=dict()):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.non_local_cfg = non_local_cfg
        self.non_local_block = NonLocal3d(self.block.conv3.norm.num_features,
                                          **self.non_local_cfg)
        self.num_segments = num_segments

    def forward(self, x):
        x = self.block(x)

        n, c, h, w = x.size()
        x = x.view(n // self.num_segments, self.num_segments, c, h,
                   w).transpose(1, 2).contiguous()
        x = self.non_local_block(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        return x


class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, net, num_segments=3, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return self.net(x)

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
        # [N, C, H, W]
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)

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

        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)

@AbstractBuildFactory.register('model')
class ResNetTSM(ResNet):
    """ResNet backbone for TSM.

    Args:
        num_segments (int): Number of frame segments. Default: 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Default: True.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages. Default: (0, 0, 0, 0).
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        shift_div (int): Number of div for shift. Default: 8.
        shift_place (str): Places in resnet layers for shift, which is chosen
            from ['block', 'blockres'].
            If set to 'block', it will apply temporal shift to all child blocks
            in each resnet layer.
            If set to 'blockres', it will apply temporal shift to each `conv1`
            layer of all child blocks in each resnet layer.
            Default: 'blockres'.
        temporal_pool (bool): Whether to add temporal pooling. Default: False.
        **kwargs (keyword arguments, optional): Arguments for ResNet.
    """

    def __init__(self,
                 depth,
                 modality="RGB",
                 clip_seg_num=8,
                 is_shift=True,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=dict(),
                 shift_div=8,
                 shift_place='blockres',
                 temporal_pool=False,
                 **kwargs):
        super().__init__(depth, **kwargs)
        self.modality = modality
        self.num_segments = clip_seg_num
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.non_local_stages = _ntuple(self.num_stages)(non_local)
        self.non_local_cfg = non_local_cfg

    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        if self.temporal_pool:
            num_segment_list = [
                self.num_segments, self.num_segments // 2,
                self.num_segments // 2, self.num_segments // 2
            ]
        else:
            num_segment_list = [self.num_segments] * 4
        if num_segment_list[-1] <= 0:
            raise ValueError('num_segment_list[-1] must be positive')

        if self.shift_place == 'block':

            def make_block_temporal(stage, num_segments):
                """Make temporal shift on some blocks.

                Args:
                    stage (nn.Module): Model layers to be shifted.
                    num_segments (int): Number of frame segments.

                Returns:
                    nn.Module: The shifted blocks.
                """
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(
                        b, num_segments=num_segments, shift_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        elif 'blockres' in self.shift_place:
            n_round = 1
            if len(list(self.layer3.children())) >= 23:
                n_round = 2

            def make_block_temporal(stage, num_segments):
                """Make temporal shift on some blocks.

                Args:
                    stage (nn.Module): Model layers to be shifted.
                    num_segments (int): Number of frame segments.

                Returns:
                    nn.Module: The shifted blocks.
                """
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1.conv = TemporalShift(
                            b.conv1.conv,
                            num_segments=num_segments,
                            shift_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        else:
            raise NotImplementedError

    def make_temporal_pool(self):
        """Make temporal pooling between layer1 and layer2, using a 3D max
        pooling layer."""

        class TemporalPool(nn.Module):
            """Temporal pool module.

            Wrap layer2 in ResNet50 with a 3D max pooling layer.

            Args:
                net (nn.Module): Module to make temporal pool.
                num_segments (int): Number of frame segments.
            """

            def __init__(self, net, num_segments):
                super().__init__()
                self.net = net
                self.num_segments = num_segments
                self.max_pool3d = nn.MaxPool3d(
                    kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

            def forward(self, x):
                # [N, C, H, W]
                n, c, h, w = x.size()
                # [N // num_segments, C, num_segments, H, W]
                x = x.view(n // self.num_segments, self.num_segments, c, h,
                           w).transpose(1, 2)
                # [N // num_segmnets, C, num_segments // 2, H, W]
                x = self.max_pool3d(x)
                # [N // 2, C, H, W]
                x = x.transpose(1, 2).contiguous().view(n // 2, c, h, w)
                return self.net(x)

        self.layer2 = TemporalPool(self.layer2, self.num_segments)

    def make_non_local(self):
        # This part is for ResNet50
        for i in range(self.num_stages):
            non_local_stage = self.non_local_stages[i]
            if sum(non_local_stage) == 0:
                continue

            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)

            for idx, non_local in enumerate(non_local_stage):
                if non_local:
                    res_layer[idx] = NL3DWrapper(res_layer[idx],
                                                 self.num_segments,
                                                 self.non_local_cfg)

    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'backbone.', r'')]):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("SVTAS")
                if self.torchvision_pretrain:
                    # torchvision's
                    self._load_torchvision_checkpoint(logger)
                else:
                    # ours
                    load_checkpoint(
                        self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)
            elif self.pretrained is None:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)
        if self.is_shift:
            self.make_temporal_shift()
        if len(self.non_local_cfg) != 0:
            self.make_non_local()
        if self.temporal_pool:
            self.make_temporal_pool()