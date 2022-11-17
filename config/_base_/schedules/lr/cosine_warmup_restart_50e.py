'''
Author       : Thyssen Wen
Date         : 2022-10-31 14:40:36
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-16 20:10:56
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/lr/cosine_warmup_restart_50e.py
'''
LRSCHEDULER = dict(
    name = "CosineAnnealingWarmupRestarts",
    first_cycle_steps=25,
    cycle_mult=1.0,
    max_lr=0.001,
    min_lr=0.0001,
    warmup_steps=10,
    gamma=0.5
)