'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:50:08
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-18 19:09:56
Description  : Swin Transformer
FilePath     : /SVTAS/config/_base_/models/action_recognition/swin_transformer.py
'''
MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "SwinTransformer3D",
        pretrained = "./data/checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth",
        pretrained2d = False,
        patch_size = [2, 4, 4],
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = [8,7,7],
        mlp_ratio = 4.,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.2,
        patch_norm = True
    )
)