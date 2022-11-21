'''
Author       : Thyssen Wen
Date         : 2022-11-21 15:08:51
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 15:19:54
Description  : file content
FilePath     : /SVTAS/config/_base_/models/image_classification/mobilenetv2.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "MobileNetV2",
        pretrained = "./data/checkpoint/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
        out_indices = (7, )
    ),
)