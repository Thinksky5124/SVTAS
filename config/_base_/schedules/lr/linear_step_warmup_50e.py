'''
Author       : Thyssen Wen
Date         : 2022-11-03 16:49:03
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 16:54:51
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/lr/linear_step_warmup_50e.py
'''
LRSCHEDULER = dict(
    name = "WarmupMultiStepLR",
    milestones=[],
    gamma=0.1,
    warmup_factor=0.3,
    warmup_iters=10 * 30,
    warmup_method='linear',
)