'''
Author       : Thyssen Wen
Date         : 2022-11-21 19:47:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 14:59:41
Description  : ref:https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/efficientnet.py
FilePath     : /SVTAS/svtas/model/backbones/utils/efficientnet/layers.py
'''
from svtas.utils.misc import is_tuple_of
from .functions import make_divisible
import torch.utils.checkpoint as cp
import torch.nn as nn
from svtas.model_pipline.wrapper import TorchBaseModel
from svtas.model_pipline.torch_utils import ConvModule, DropPath
        
class InvertedResidual(TorchBaseModel):
    """Inverted Residual Block.
    Args:
        in_channels (int): The input channels of this module.
        out_channels (int): The output channels of this module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution.
            Defaults to 3.
        stride (int): The stride of the depthwise convolution. Defaults to 1.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.with_expand_conv = (mid_channels != in_channels)

        if self.with_se:
            assert isinstance(se_cfg, dict)

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.with_se:
            self.se = SqueezeExcitation(**se_cfg)
        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """

        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            out = self.depthwise_conv(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                return x + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

class SqueezeExcitation(TorchBaseModel):
    """Squeeze-and-Excitation Module.
    Args:
        channels (int): The input (and output) channels of the SE layer.
        squeeze_channels (None or int): The intermediate channel number of
            SqueezeExcitation. Default: None, means the value of ``squeeze_channels``
            is ``make_divisible(channels // ratio, divisor)``.
        ratio (int): Squeeze ratio in SqueezeExcitation, the intermediate channel will
            be ``make_divisible(channels // ratio, divisor)``. Only used when
            ``squeeze_channels`` is None. Default: 16.
        divisor(int): The divisor to true divide the channel number. Only
            used when ``squeeze_channels`` is None. Default: 8.
        conv_cfg (None or dict): Config dict for convolution layer. Default:
            None, which means using conv2d.
        return_weight(bool): Whether to return the weight. Default: False.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 squeeze_channels=None,
                 ratio=16,
                 divisor=8,
                 bias='auto',
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 return_weight=False,
                 init_cfg=None):
        super(SqueezeExcitation, self).__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if squeeze_channels is None:
            squeeze_channels = make_divisible(channels // ratio, divisor)
        assert isinstance(squeeze_channels, int) and squeeze_channels > 0, \
            '"squeeze_channels" should be a positive integer, but get ' + \
            f'{squeeze_channels} instead.'
        self.return_weight = return_weight
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=squeeze_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        if self.return_weight:
            return out
        else:
            return x * out
            
class EdgeResidual(TorchBaseModel):
    """Edge Residual Block.
    Args:
        in_channels (int): The input channels of this module.
        out_channels (int): The output channels of this module.
        mid_channels (int): The input channels of the second convolution.
        kernel_size (int): The kernel size of the first convolution.
            Defaults to 3.
        stride (int): The stride of the first convolution. Defaults to 1.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        with_residual (bool): Use residual connection. Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 with_residual=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None):
        super(EdgeResidual, self).__init__(init_cfg=init_cfg)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.with_residual = (
            stride == 1 and in_channels == out_channels and with_residual)

        if self.with_se:
            assert isinstance(se_cfg, dict)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.with_se:
            self.se = SqueezeExcitation(**se_cfg)

        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):

        def _inner_forward(x):
            out = x
            out = self.conv1(out)

            if self.with_se:
                out = self.se(out)

            out = self.conv2(out)

            if self.with_residual:
                return x + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out