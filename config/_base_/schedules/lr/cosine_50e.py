'''
Author       : Thyssen Wen
Date         : 2022-10-31 14:46:11
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:50:01
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/lr/cosine_50e.py
'''
LRSCHEDULER = dict(
    name = "CosineAnnealingLR",
    T_max = [50]
)