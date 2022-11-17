'''
Author       : Thyssen Wen
Date         : 2022-11-16 16:37:20
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-16 20:05:52
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/lr/multistep_warmup_50e.py
'''
LRSCHEDULER = dict(
    name = "WarmupMultiStepLR",
    milestones=[20, 35],
    gamma=0.1,
    warmup_factor=0.3,
    warmup_iters=0,
    warmup_method='constant'
)