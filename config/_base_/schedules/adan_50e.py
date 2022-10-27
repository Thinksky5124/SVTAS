'''
Author       : Thyssen Wen
Date         : 2022-10-27 10:39:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 10:45:57
Description  : Adan 50 epoches config
FilePath     : /SVTAS/config/_base_/schedules/adan_50e.py
'''
epochs = 50
OPTIMIZER = dict(
    name = "AdanOptimizer",
    learning_rate = 1e-3,
    weight_decay = 0.02,
    betas = (0.9, 0.999)
)

LRSCHEDULER = dict(
    name = "MultiStepLR",
    step_size = [epochs],
    gamma = 0.1
)