'''
Author       : Thyssen Wen
Date         : 2023-02-25 19:52:48
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-26 15:11:16
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/transformer_xl.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "TransformerXL",
        in_channels=512,
        num_classes=11,
        sample_rate=1,
        n_layer=4,
        n_head=2,
        d_model=200,
        d_head=2,
        dropout=0.0,
        dropatt=0.0,
        pre_lnorm=False,
        tgt_len=64,
        ext_len=0,
        mem_len=128,
        attn_type=0,
        clamp_len=-1
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)