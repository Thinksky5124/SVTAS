'''
Author       : Thyssen Wen
Date         : 2022-11-22 21:00:58
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-22 21:04:44
Description  : file content
FilePath     : /SVTAS/config/_base_/models/action_recognition/resnet_tsm.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "ResNetTSM",
        pretrained = "./data/checkpoint/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth",
        depth=50,
        clip_seg_num = 8,
        shift_div = 8,
        norm_eval=False,
        torchvision_pretrain=False,
    )
)