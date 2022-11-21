'''
Author       : Thyssen Wen
Date         : 2022-11-21 16:33:42
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 16:45:00
Description  : file content
FilePath     : /SVTAS/config/_base_/models/image_classification/resnet.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        torchvision_pretrain=True,
        pretrained="./data/checkpoint/resnet50-0676ba61.pth",
    ),
)