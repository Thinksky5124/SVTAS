'''
Author       : Thyssen Wen
Date         : 2023-02-25 19:53:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-28 14:57:10
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/block_recurrent_transformer.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "BRTSegmentationHead",
        num_head=1,
        dim_head=1,
        state_len=512,
        causal=False,
        num_decoders=3,
        num_layers=10,
        num_f_maps=64,
        input_dim=2048,
        num_classes=11,
        channel_masking_rate=0.5,
        sample_rate=1,
        out_feature=False
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)