'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:55:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-17 14:05:47
Description  : Stream Collater Config
FilePath     : /SVTAS/config/_base_/collater/stream_compose.py
'''
COLLATE = dict(
    train=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num"]
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