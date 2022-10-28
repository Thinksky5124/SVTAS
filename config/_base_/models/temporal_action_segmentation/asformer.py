'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:58:48
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:11:59
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/asformer.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "ASFormer",
        num_decoders = 2,
        num_layers = 4,
        r1 = 2,
        r2 = 2,
        num_f_maps = 64,
        input_dim = 768,
        num_classes = 11,
        sample_rate = 1,
        channel_masking_rate = 0.5
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)