'''
Author       : Thyssen Wen
Date         : 2022-10-31 14:40:36
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:53:18
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/lr/cosine_warmup_50e.py
'''
LRSCHEDULER = dict(
    name = "CosineAnnealingWarmupRestarts",
    first_cycle_steps=50,
    cycle_mult=1.0,
    max_lr=0.1,
    min_lr=0.001,
    warmup_steps=10,
    gamma=1.0
)