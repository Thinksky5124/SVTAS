'''
Author       : Thyssen Wen
Date         : 2022-10-28 10 = 59 = 31
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 15:20:28
Description  : SLViT
FilePath     : /SVTAS/config/_base_/models/image_classification/slvit.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "SLViT",
        image_size = 224,
        patch_size = 32,
        depth = 4,
        heads = 12,
        mlp_dim = 1024,
        dropout = 0.3,
        emb_dropout = 0.3,
    ),
)