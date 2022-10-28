'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:47:54
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:48:03
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/sgd_100e.py
'''
epochs = 100
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