'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:55:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 15:09:53
Description  : Stream Collater Config
FilePath     : /SVTAS/config/_base_/collater/stream_compose.py
'''
COLLATE = dict(
    name = "StreamBatchCompose",
    to_tensor_keys = ["imgs", "labels", "masks", "precise_sliding_num"]
)