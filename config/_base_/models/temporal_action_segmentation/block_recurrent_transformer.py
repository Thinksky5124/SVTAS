'''
Author       : Thyssen Wen
Date         : 2023-02-25 19:53:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-25 19:54:16
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/block_recurrent_transformer.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "BRTSegmentationHead",
        in_channels=512,
        hidden_channels=512,
        num_classes=11,
        num_head=1,
        dim_head=1,
        sample_rate=1,
        state_len=512,
        causal=False,
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)