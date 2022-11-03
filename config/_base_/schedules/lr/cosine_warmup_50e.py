'''
Author       : Thyssen Wen
Date         : 2022-11-03 16:49:23
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 16:53:23
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/lr/cosine_warmup_50e.py
'''
LRSCHEDULER = dict(
    name = "WarmupCosineLR",
    max_iters=50 * 30,
    warmup_factor=0.3,
    warmup_iters=10.0 * 30,
    warmup_method='linear',
)