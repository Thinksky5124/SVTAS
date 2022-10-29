'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:19:28
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 16:19:40
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/adamw_100e.py
'''
epochs = 100 #Mandatory, total epoch
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