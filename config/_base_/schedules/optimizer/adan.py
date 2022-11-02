'''
Author       : Thyssen Wen
Date         : 2022-10-27 10:39:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:49:43
Description  : Adan 50 epoches config
FilePath     : /SVTAS/config/_base_/schedules/optimizer/adan.py
'''
OPTIMIZER = dict(
    name = "AdanOptimizer",
    learning_rate = 1e-3,
    weight_decay = 0.02,
    betas = (0.98, 0.92, 0.99)
)