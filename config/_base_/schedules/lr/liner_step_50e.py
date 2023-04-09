'''
Author       : Thyssen Wen
Date         : 2022-10-31 14:40:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-03-26 12:21:50
Description  : Linear_step
FilePath     : /SVTAS/config/_base_/schedules/lr/liner_step_50e.py
'''
LRSCHEDULER = dict(
    name = "MultiStepLR",
    step_size = [50],
    gamma = 0.1
)