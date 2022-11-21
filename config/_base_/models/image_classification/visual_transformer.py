'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:59:40
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 19:25:37
Description  : ViT
FilePath     : /SVTAS/config/_base_/models/image_classification/visual_transformer.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "VisionTransformer",
        pretrained = "./data/checkpoint/vit_b_16-c867db91.pth",
        image_size = 224,
        patch_size = 16,
        num_layers = 12,
        num_heads = 12,
        hidden_dim = 768,
        mlp_dim = 3072,
    ),    
)