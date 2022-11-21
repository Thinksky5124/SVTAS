'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:59:40
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 15:20:33
Description  : ViT
FilePath     : /SVTAS/config/_base_/models/image_classification/vit.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "ViT",
        image_size = 224,
        patch_size = 32,
        depth = 4,
        heads = 12,
        mlp_dim = 1024,
        dropout = 0.3,
        emb_dropout = 0.3
    ),    
)