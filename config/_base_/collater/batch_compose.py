'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:37:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 16:52:38
Description  : Compse Funtion Config
FilePath     : /SVTAS/config/_base_/collater/batch_compose.py
'''
COLLATE = dict(
    name = "BatchCompose",
    to_tensor_keys = ["imgs", "feature", "labels", "masks", "precise_sliding_num"],
    compress_keys = ["sliding_num", "current_sliding_cnt"],
    ignore_index = -100
)