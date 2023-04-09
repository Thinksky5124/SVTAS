'''
Author       : Thyssen Wen
Date         : 2022-10-30 13:51:49
LastEditors  : Thyssen Wen
LastEditTime : 2023-03-31 09:20:02
Description  : MViT config
FilePath     : /SVTAS/config/_base_/models/action_recognition/mvitv2_b.py
'''
MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "MViT"
    )
)

