'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:37:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 20:12:42
Description  : Compse Funtion Config
FilePath     : /SVTAS/config/_base_/dataloader/collater/batch_compose.py
'''
COLLATE = dict(
    train = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        dropout_keys = [""],
        compress_keys = ["sliding_num", "current_sliding_cnt", "step"],
        ignore_index = -100
    ),
    test = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        dropout_keys = [""],
        compress_keys = ["sliding_num", "current_sliding_cnt", "step"],
        ignore_index = -100
    ),
    infer = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        dropout_keys = [""],
        compress_keys = ["sliding_num", "current_sliding_cnt", "step"],
        ignore_index = -100
    )
)