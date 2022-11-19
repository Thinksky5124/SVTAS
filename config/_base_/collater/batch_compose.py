'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:37:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-17 14:06:00
Description  : Compse Funtion Config
FilePath     : /SVTAS/config/_base_/collater/batch_compose.py
'''
COLLATE = dict(
    train = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        compress_keys = ["sliding_num", "current_sliding_cnt"],
        ignore_index = -100
    ),
    test = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        compress_keys = ["sliding_num", "current_sliding_cnt"],
        ignore_index = -100
    ),
    infer = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        compress_keys = ["sliding_num", "current_sliding_cnt"],
        ignore_index = -100
    )
)