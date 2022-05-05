'''
Author: Thyssen Wen
Date: 2022-04-16 16:40:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-04 16:16:44
Description: caculate model flops and param
FilePath     : /ETESVS/tools/caculate_model_flops_param.py
'''
import torch
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
from torchinfo import summary
from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from thop import clever_format
from model.backbones.i3d import InceptionI3d

# I3D model param flops caculate
clip_seg_num = 64
sample_rate = 4
x_shape = [2, clip_seg_num, 244, 244]
mask_shape = [clip_seg_num * sample_rate]
input_shape = (x_shape, mask_shape)
def input_constructor(input_shape):
    x_shape, mask_shape = input_shape
    x = torch.randn([1] + x_shape).cuda()
    mask = torch.randn([1] + mask_shape).cuda()
    idx = torch.randn([1] + [1]).cuda()
    return dict(imgs=x, masks=mask, idx=idx)
output = input_constructor(input_shape)
x, mask = output["imgs"], output["masks"]
model = InceptionI3d(num_classes=11, in_channels=2)
summary(model, input_size=[x.shape], col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
print("="*20)
print('Use mmcv get_model_complexity_info function')
flops_number, params_number = get_model_complexity_info(model, input_shape=(2, clip_seg_num, 244, 244), print_per_layer_stat=False, as_strings=False)
flops_per_image_number = flops_number / clip_seg_num
flops, params = clever_format([flops_number, params_number], "%.6f")
flops_per_image, params = clever_format([flops_per_image_number, params_number], "%.6f")
print("Hitp: This FLOPs is caculation by", clip_seg_num, "imgs")
print("Per Image FLOPs:", flops_per_image, ", Total FLOPs:", flops, ", Total params", params)
print("="*20)