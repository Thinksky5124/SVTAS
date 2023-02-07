'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:58:56
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-07 13:39:52
Description  : LinFormer
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/linformer.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    head = dict(
        name ="LinformerHead",
        num_decoders = 3,
        num_layers = 10,
        num_f_maps = 64,
        input_dim = 2048,
        num_classes = 11,
        sample_rate = 1,
        channel_masking_rate = 0.5
    )
)