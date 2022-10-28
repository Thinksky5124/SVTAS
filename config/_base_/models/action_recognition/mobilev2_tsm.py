'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:49:54
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 10:51:05
Description  : MobileNetV2 TSM
FilePath     : /SVTAS/config/_base_/models/action_recognition/mobilev2_tsm.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "MobileNetV2TSM",
        pretrained = "./data/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth",
        clip_seg_num = 8,
        shift_div = 8,
        out_indices = (7, )
    ),
    neck = None,
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 1280,
        input_seg_num = 8,
        output_seg_num = 1,
        sample_rate = 8,
        pool_space = True,
        in_format = "N*T,C,H,W",
        out_format = "NCT"
    ),
    loss = None
)