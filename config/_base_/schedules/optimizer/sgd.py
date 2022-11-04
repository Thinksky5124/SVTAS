'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:47:45
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 15:29:01
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/optimizer/sgd.py
'''
OPTIMIZER = dict(
    name = "SGDOptimizer",
    learning_rate = 1e-3,
    weight_decay = 0.02,
    need_grad_accumulate = True
)