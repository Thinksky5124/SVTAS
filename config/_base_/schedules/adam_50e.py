'''
Author       : Thyssen Wen
Date         : 2022-10-25 15:59:58
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-25 16:07:56
Description  : Optimizer Config
FilePath     : /SVTAS/config/_base_/schedules/adam_50e.py
'''
from math import gamma
from unicodedata import name


OPTIMIZER = dict(
    name = "AdamOptimizer",
    learning_rate = 0.0005,
    weight_decay = 1e-4,
    betas = (0.9, 0.999)
)

LRSCHEDULER = dict(
    name = "MultiStepLR",
    step_size = [50],
    gamma = 0.1
)