'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:50:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-19 13:19:43
Description  : Timesformer
FilePath     : /SVTAS/config/_base_/models/action_recognition/timesformer.py
'''
MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "TimeSformer",
        pretrained = "./data/checkpoint/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth",
        num_frames = 8,
        img_size = 224,
        patch_size = 16,
        embed_dims = 768
    ),
    neck = None,
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 768,
        input_seg_num = 8,
        output_seg_num = 1,
        sample_rate = 8,
        pool_space = True,
        in_format = "N*T,C",
        out_format = "NCT"
    ),
    loss = None
)