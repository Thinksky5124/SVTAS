'''
Author       : Thyssen Wen
Date         : 2022-10-31 14:40:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:46:16
Description  : Linear_step
FilePath     : /SVTAS/config/_base_/schedules/lr/liner_setp.py
'''
LRSCHEDULER = dict(
    name = "MultiStepLR",
    step_size = [50],
    gamma = 0.1
)