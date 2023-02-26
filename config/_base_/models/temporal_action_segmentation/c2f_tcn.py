'''
Author       : Thyssen Wen
Date         : 2023-02-24 15:17:10
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-24 15:32:44
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/c2f_tcn.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "C2F_TCN",
        n_channels = 512,
        num_classes = 11,
        ensem_weights = [1, 1, 1, 1, 0, 0],
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