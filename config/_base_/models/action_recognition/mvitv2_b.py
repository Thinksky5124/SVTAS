'''
Author       : Thyssen Wen
Date         : 2022-10-30 13:51:49
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-30 15:48:59
Description  : MViT config
FilePath     : /SVTAS/config/_base_/models/action_recognition/mvitv2_b.py
'''
MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "MViT"
    ),
    neck = None,
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 1024,
        input_seg_num = 8,
        output_seg_num = 1,
        sample_rate = 1,
        pool_space = True,
        in_format = "N,C,T,H,W",
        out_format = "NCT"
    ),
    loss = None
)

