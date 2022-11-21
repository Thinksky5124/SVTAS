'''
Author       : Thyssen Wen
Date         : 2022-11-21 20:15:20
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 20:37:37
Description  : file content
FilePath     : /SVTAS/config/_base_/models/image_classification/swin_v2_transformer.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "SwinTransformerV2",
        pretrained = "./data/checkpoint/swinv2_tiny_patch4_window8_256.pth",
        img_size=256,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        drop_path_rate=0.2,
    ),    
)