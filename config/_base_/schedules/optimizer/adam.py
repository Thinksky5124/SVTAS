'''
Author       : Thyssen Wen
Date         : 2022-10-25 15:59:58
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-12 10:36:56
Description  : Optimizer Config
FilePath     : /SVTAS/config/_base_/schedules/optimizer/adam.py
'''
OPTIMIZER = dict(
    name = "AdamOptimizer",
    learning_rate = 0.0005,
    weight_decay = 1e-4,
    betas = (0.9, 0.999),
    need_grad_accumulate = True,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = [],
    freeze_key = [],
)