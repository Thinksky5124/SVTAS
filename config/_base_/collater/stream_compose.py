'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:55:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-25 16:55:06
Description  : Stream Collater Config
FilePath     : /SVTAS/config/_base_/collater/stream_collater.py
'''
COLLATE = dict(
    name = "StreamBatchCompose",
    to_tensor_keys = ["imgs", "labels", "masks"]
)