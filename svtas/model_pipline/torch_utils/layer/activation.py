'''
Author       : Thyssen Wen
Date         : 2023-10-11 19:17:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 19:24:40
Description  : ref: https://github.com/open-mmlab/mmcv/blob/57c4e25e06e2d4f8a9357c84bcd24089a284dc88/mmcv/cnn/bricks/activation.py
FilePath     : /SVTAS/svtas/model_pipline/torch_utils/layer/activation.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from svtas.utils import AbstractBuildFactory
from svtas.utils.package_utils import digit_version

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh, nn.GELU
]:
    AbstractBuildFactory.register_obj(obj=module, registory_name='model')

if digit_version(torch.__version__) >= digit_version('1.7.0'):
    AbstractBuildFactory.register_obj(obj=nn.SiLU, registory_name='model', obj_name='SiLU')
else:

    class SiLU(nn.Module):
        """Sigmoid Weighted Liner Unit."""

        def __init__(self, inplace=False):
            super().__init__()
            self.inplace = inplace

        def forward(self, inputs) -> torch.Tensor:
            if self.inplace:
                return inputs.mul_(torch.sigmoid(inputs))
            else:
                return inputs * torch.sigmoid(inputs)

    AbstractBuildFactory.register_obj(obj=SiLU, registory_name='model', obj_name='SiLU')
