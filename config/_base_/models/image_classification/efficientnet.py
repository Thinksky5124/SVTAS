'''
Author       : Thyssen Wen
Date         : 2022-11-21 16:16:58
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 16:23:30
Description  : file content
FilePath     : /SVTAS/config/_base_/models/image_classification/efficientnet.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "EfficientNet",
        arch='b1',
        pretrained="data/checkpoint/efficientnet-b1_3rdparty_8xb32_in1k_20220119-002556d9.pth",
    ),
)