'''
Author       : Thyssen Wen
Date         : 2022-11-17 14:03:54
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-17 14:06:13
Description  : file content
FilePath     : /SVTAS/config/_base_/collater/batch_stream_compose.py
'''
COLLATE = dict(
    train = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        compress_keys = ["sliding_num", "current_sliding_cnt"],
        ignore_index = -100
    ),
    test=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num"]
    ),
    infer=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num"]
    )
)