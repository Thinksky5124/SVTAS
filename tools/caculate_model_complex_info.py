'''
Author: Thyssen Wen
Date: 2022-04-16 16:40:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-13 14:06:50
Description: caculate model flops param infer-time fps and throughput
FilePath     : /ETESVS/tools/caculate_model_complex_info.py
'''
import torch
import os
import sys
import numpy as np
path = os.path.join(os.getcwd())
sys.path.append(path)
from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, flop_count_table
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
dummy_input = input_constructor(input_shape)
model = InceptionI3d(num_classes=11, in_channels=2)
# mmcv caculate param and flops
print("="*20)
print('Use mmcv get_model_complexity_info function')
flops_number, params_number = get_model_complexity_info(model, input_shape=input_shape, input_constructor=input_constructor, print_per_layer_stat=False, as_strings=False)
flops_per_image_number = flops_number / cfg.DATASET.test.clip_seg_num
flops, params = clever_format([flops_number, params_number], "%.6f")
flops_per_image, params = clever_format([flops_per_image_number, params_number], "%.6f")
print("Hitp: This FLOPs is caculation by {clip_seg_num:d} imgs".format(clip_seg_num=cfg.DATASET.test.clip_seg_num))
print("Per Image FLOPs:"+ flops_per_image + ", Total FLOPs:" + flops + ", Total params:" + params)
print("="*20)

# fvcore caculate param and flops
print('Use fvcore FlopCountAnalysis function')
inputs = (dummy_input['input_data'])
flops = FlopCountAnalysis(model, inputs)
print(flop_count_table(flops))
flops_number = flops.total()
flops_per_image_number = flops_number / cfg.DATASET.test.clip_seg_num
flops = clever_format([flops_number], "%.6f")
flops_per_image = clever_format([flops_per_image_number], "%.6f")
print("Hitp: This FLOPs is caculation by {clip_seg_num:d} imgs".format(clip_seg_num=cfg.DATASET.test.clip_seg_num))
print("Per Image FLOPs:"+ flops_per_image + ", Total FLOPs:" + flops)
print("="*20)

# model fps caculate
dummy_input = dummy_input['input_data']
print('Caculate model fps (single frame infer times)')
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))

#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn * clip_seg_num
print('Mean@1 {mean_syn:.3f}ms, Std@5 {std_syn:.3f}ms, FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print('Model single forward infer time(ms) {mean_syn:.3f}ms'.format(mean_syn=mean_syn))
print("="*20)

# model latency time
print('Caculate model Throughput')
repetitions=100
total_time = 0
# it should be modify by every model
optimal_batch_size=1
dummy_input = input_constructor(input_shape, optimal_batch_size=optimal_batch_size)['input_data']
with torch.no_grad():
    for rep in range(repetitions):
        starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) / 1000
        total_time += curr_time
Throughput = (repetitions * optimal_batch_size) / total_time
print("Final Throughput: {Throughput:.2f} V/s, Measuring by batch_size: {Batch_size:d}".format(Throughput=Throughput, Batch_size=optimal_batch_size))
print("="*20)