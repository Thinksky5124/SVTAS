'''
Author       : Thyssen Wen
Date         : 2023-02-08 11:38:31
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-08 14:52:42
Description  : ASRF Model
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/asrf.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    head = dict(
        name = "ActionSegmentRefinementFramework",
        in_channel = 2048,
        num_features = 64,
        num_stages = 4,
        num_layers = 10,
        num_classes = 11,
        sample_rate = 1
    ),
    loss = dict(
        name = "ASRFLoss",
        num_classes = 11,
        sample_rate = 1,
        ignore_index = -100
    )
)