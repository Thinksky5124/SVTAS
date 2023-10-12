'''
Author       : Thyssen Wen
Date         : 2022-11-02 11:07:12
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-02 16:05:12
Description  : X3D model ref:https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/backbones/x3d.py
FilePath     : /SVTAS/svtas/model/backbones/video/x3d.py
'''
import torch
from torch import nn
import torch.nn.functional as F
import math
from ....utils.logger import get_logger
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline.torch_utils import load_state_dict, c2_msra_fill, c2_xavier_fill
from ..utils import get_norm, VideoModelStem, ResStage
from ..utils import (round_width)

@AbstractBuildFactory.register('model')
class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.
    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self,
                 pretrained=None,
                 norm_type="batchnorm",
                 dim_c1=12,
                 scale_res2=False,
                 depth=50,
                 num_groups=1,
                 width_per_group=64,
                 width_factor=1.0,
                 depth_factor=1.0,
                 input_channel_num=[3],
                 bottleneck_factor=1.0,
                 channelwise_3x3x3=True,
                 nonlocal_location=[[[]], [[]], [[]], [[]]],
                 nonlocal_group=[[1], [1], [1], [1]],
                 nonlocal_pool=[# Res2
                                 [[1, 2, 2], [1, 2, 2]],
                                 # Res3
                                 [[1, 2, 2], [1, 2, 2]],
                                 # Res4
                                 [[1, 2, 2], [1, 2, 2]],
                                 # Res5
                                 [[1, 2, 2], [1, 2, 2]],],
                 nonlocal_instantition="dot_product",
                 stride_1x1=False,
                 spatial_dilations=[[1], [1], [1], [1]],
                 dropcounnect_rate=0.0,
                 fc_init_std = 0.01,
                 zero_init_final_bn = True,
                 zero_init_final_conv = False):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.pretrained = pretrained
        self.fc_init_std = fc_init_std
        self.zero_init_final_bn = zero_init_final_bn
        self.zero_init_final_conv = zero_init_final_conv

        self.norm_module = get_norm(norm_type)
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = dim_c1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if scale_res2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(depth=depth,
                           num_groups=num_groups,
                           width_per_group=width_per_group,
                           width_factor=width_factor,
                           depth_factor=depth_factor,
                           input_channel_num=input_channel_num,
                           bottleneck_factor=bottleneck_factor,
                           channelwise_3x3x3=channelwise_3x3x3,
                           nonlocal_location=nonlocal_location,
                           nonlocal_group=nonlocal_group,
                           nonlocal_pool=nonlocal_pool,
                           nonlocal_instantition=nonlocal_instantition,
                           stride_1x1=stride_1x1,
                           spatial_dilations=spatial_dilations,
                           dropcounnect_rate=dropcounnect_rate)
    
    def init_weights(self, child_model=False, revise_keys=[(r'backbone.', r'')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger  = get_logger("SVTAS")
                checkpoint = torch.load(self.pretrained)
                load_state_dict(self, checkpoint['model_state'], strict=False, logger=logger)
            else:
                self._init_weights()
        else:
            self._init_weights()
    
    def _clear_memory_buffer(self):
        pass
        
    def _init_weights(self):
        """
        Performs ResNet style weight initialization.
        Args:
            fc_init_std (float): the expected standard deviation for fc layer.
            zero_init_final_bn (bool): if True, zero initialize the final bn for
                every bottleneck.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Note that there is no bias due to BN
                if hasattr(m, "final_conv") and self.zero_init_final_conv:
                    m.weight.data.zero_()
                else:
                    """
                    Follow the initialization method proposed in:
                    {He, Kaiming, et al.
                    "Delving deep into rectifiers: Surpassing human-level
                    performance on imagenet classification."
                    arXiv preprint arXiv:1502.01852 (2015)}
                    """
                    c2_msra_fill(m)

            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if (
                    hasattr(m, "transform_final_bn")
                    and m.transform_final_bn
                    and self.zero_init_final_bn
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                if m.weight is not None:
                    m.weight.data.fill_(batchnorm_weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                if hasattr(m, "xavier_init") and m.xavier_init:
                    c2_xavier_fill(m)
                else:
                    m.weight.data.normal_(mean=0.0, std=self.fc_init_std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self,
                           depth=50,
                           num_groups=1,
                           width_per_group=64,
                           width_factor=1.0,
                           depth_factor=1.0,
                           input_channel_num=[3,3],
                           bottleneck_factor=1.0,
                           channelwise_3x3x3=True,
                           nonlocal_location=[[[]], [[]], [[]], [[]]],
                           nonlocal_group=[[1], [1], [1], [1]],
                           nonlocal_pool=[# Res2
                                          [[1, 2, 2], [1, 2, 2]],
                                          # Res3
                                          [[1, 2, 2], [1, 2, 2]],
                                          # Res4
                                          [[1, 2, 2], [1, 2, 2]],
                                          # Res5
                                          [[1, 2, 2], [1, 2, 2]],],
                           nonlocal_instantition="dot_product",
                           stride_1x1=False,
                           spatial_dilations=[[1], [1], [1], [1]],
                           dropcounnect_rate=0.0):
        """
        Builds a single pathway X3D model.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        _MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}
        assert depth in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[depth]
        assert width_per_group % num_groups == 0
        dim_inner = num_groups * width_per_group

        w_mul = width_factor
        d_mul = depth_factor
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = [
                    [[5]],  # conv1 temporal kernels.
                    [[3]],  # res2 temporal kernels.
                    [[3]],  # res3 temporal kernels.
                    [[3]],  # res4 temporal kernels.
                    [[3]],  # res5 temporal kernels.
                ]

        self.s1 = VideoModelStem(
            dim_in=input_channel_num,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(bottleneck_factor * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if channelwise_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=nonlocal_location[0],
                nonlocal_group=nonlocal_group[0],
                nonlocal_pool=nonlocal_pool[0],
                instantiation=nonlocal_instantition,
                trans_func_name="x3d_transform",
                stride_1x1=stride_1x1,
                norm_module=self.norm_module,
                dilation=spatial_dilations[stage],
                drop_connect_rate=dropcounnect_rate
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

    def forward(self, x, masks):
        x = x.unsqueeze(0)
        for module in self.children():
            x = module(x)
        masks = F.adaptive_max_pool3d(masks, output_size=[x[0].shape[2], 1, 1])
        return x[0] * masks