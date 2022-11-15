'''
Author       : Thyssen Wen
Date         : 2022-11-15 09:55:23
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-15 10:02:34
Description  : file content
FilePath     : /SVTAS/config/_base_/models/action_recognition/i3d_r50.py
'''
MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "ResNet3d",
        pretrained = "./data/checkpoint/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth",
        in_channels=3,
        pretrained2d=False,
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False,
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