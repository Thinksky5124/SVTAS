'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:19:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-12 10:37:50
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/optimizer/adamw.py
'''
OPTIMIZER = dict(
    name = "AdamWOptimizer",
    learning_rate = 0.0005,
    weight_decay = 1e-4,
    betas = (0.9, 0.999),
    need_grad_accumulate = True,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = [],
    freeze_key = [],
)