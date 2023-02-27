'''
Author       : Thyssen Wen
Date         : 2023-02-08 11:38:31
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-26 20:43:22
Description  : ASRF Model
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/asrf_brt.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    head = dict(
        name = "ASRFWithBRT",
        in_channel = 2048,
        num_features = 64,
        num_classes = 11, 
        num_stages = 4,
        num_head = 1,
        dim_head = 128,
        sample_rate = 1,
        state_len = 512,
        num_layers = 5,
        causal = False
    ),
    loss = dict(
        name = "ASRFLoss",
        num_classes = 11,
        sample_rate = 1,
        ignore_index = -100
    )
)