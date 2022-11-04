'''
Author       : Thyssen Wen
Date         : 2022-11-03 20:56:34
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 19:39:03
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/conformer.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "Conformer",
        num_classes = 11,
        sample_rate = 1,
        input_dim = 2048,
        encoder_dim = 64,
        num_encoder_layers = 8,
        input_dropout_p = 0.5,
        num_attention_heads = 8,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 31,
        half_step_residual = True
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)