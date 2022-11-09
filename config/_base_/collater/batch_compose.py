'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:37:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-09 14:50:19
Description  : Compse Funtion Config
FilePath     : /SVTAS/config/_base_/collater/batch_compose.py
'''
COLLATE = dict(
    name = "BatchCompose",
    to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
    compress_keys = ["sliding_num", "current_sliding_cnt"],
    ignore_index = -100
)