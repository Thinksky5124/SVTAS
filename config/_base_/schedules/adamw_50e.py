'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:19:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 16:19:42
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/adamw_50e.py
'''
epochs = 50 #Mandatory, total epoch
OPTIMIZER = dict(
    name = "AdamWOptimizer",
    learning_rate = 0.0005,
    weight_decay = 1e-4,
    betas = (0.9, 0.999)
)

LRSCHEDULER = dict(
    name = "MultiStepLR",
    step_size = [epochs],
    gamma = 0.1
)