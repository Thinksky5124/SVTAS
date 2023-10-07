'''
Author       : Thyssen Wen
Date         : 2022-10-27 18:23:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 14:14:15
Description  : MS-TCN model
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/ms_tcn.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "MultiStageModel",
        num_stages = 4,
        num_layers = 10,
        num_f_maps = 64,
        dim = 512,
        num_classes = 11,
        sample_rate = 1
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)
