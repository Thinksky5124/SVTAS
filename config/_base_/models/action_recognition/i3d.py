'''
Author       : Thyssen Wen
Date         : 2022-10-27 19:16:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 19:16:54
Description  : I3D model
FilePath     : /SVTAS/config/_base_/models/action_recognition/i3d.py
'''
MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "I3D",
        pretrained = "./data/checkpoint/i3d_rgb.pt",
        in_channels = 3
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