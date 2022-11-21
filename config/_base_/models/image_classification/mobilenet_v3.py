'''
Author       : Thyssen Wen
Date         : 2022-11-21 19:16:01
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 19:17:58
Description  : file content
FilePath     : /SVTAS/config/_base_/models/image_classification/mobilenet_v3.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "MobileNetV3",
        pretrained = "./data/checkpoint/mobilenet_v3_large-3ea3c186.pth",
        out_indices = (7, )
    ),
)