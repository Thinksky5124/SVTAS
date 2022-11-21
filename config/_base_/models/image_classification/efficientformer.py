'''
Author       : Thyssen Wen
Date         : 2022-11-21 14:00:29
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 15:20:24
Description  : file content
FilePath     : /SVTAS/config/_base_/models/image_classification/efficientformer.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "EfficientFormer",
        arch='l1',
        pretrained="data/checkpoint/efficientformer-l1_3rdparty_in1k_20220803-d66e61df.pth",
        reshape_last_feat=True,
        drop_path_rate=0,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
            dict(type='Constant', layer=['LayerScale'], val=1e-5)
        ]
    ),
)