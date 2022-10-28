'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:47:45
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:48:22
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/sgd_50e.py
'''
epochs = 50
OPTIMIZER = dict(
    name = "SGDOptimizer",
    learning_rate = 1e-3,
    weight_decay = 0.02
)

LRSCHEDULER = dict(
    name = "MultiStepLR",
    step_size = [epochs],
    gamma = 0.1
)