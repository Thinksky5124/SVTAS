'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:59:40
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 15:00:18
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
    neck = None,
    head = dict(
        name = "TimeSformerHead",
        num_classes = 11,
        clip_seg_num = 8,
        sample_rate = 4,
        in_channels = 1024
    ),
    loss = dict(
        name = "RecognitionSegmentationLoss",
        label_mode = "hard",
        num_classes = 11,
        sample_rate = 4,
        loss_weight = 1.0,
        ignore_index = -100
    )       
)