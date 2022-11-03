'''
Author       : Thyssen Wen
Date         : 2022-11-02 14:21:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-02 21:26:58
Description  : file content
FilePath     : /SVTAS/config/_base_/models/action_recognition/x3d_m.py
'''
MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "X3D",
        pretrained="data/checkpoint/x3d_m.pyth",
        dim_c1=12,
        scale_res2=False,
        depth=50,
        num_groups=1,
        width_per_group=64,
        width_factor=2.0,
        depth_factor=2.2,
        input_channel_num=[3],
        bottleneck_factor=2.25,
        channelwise_3x3x3=True
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