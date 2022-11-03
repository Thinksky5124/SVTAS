'''
Author       : Thyssen Wen
Date         : 2022-10-27 18:23:16
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 13:01:12
Description  : MS-TCN model
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/3d_tcn.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation3D",
    backbone = None,
    neck = None,
    head = dict(
        name = "TCN3DHead",
        seg_in_channels = 768,
        num_layers = 10,
        num_f_maps = 64,
        num_classes = 11,
        sample_rate = 1,
        num_stages = 4
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)
