'''
Author       : Thyssen Wen
Date         : 2022-11-03 20:00:51
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 14:43:15
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/lstr.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "LSTR",
        modality='twostream',
        visual_size=1024,
        motion_size=1024,
        linear_enabled=True,
        linear_out_features=1024,
        long_memory_num_samples=512,
        work_memory_num_samples=32,
        num_heads=16,
        dim_feedforward=1024,
        dropout=0.2,
        activation='relu',
        num_classes=11,
        enc_module=[
                [16, 1, True], [32, 2, True]
                ],
        dec_module=[-1, 2, True],
        sample_rate=1
    ),
    loss = dict(
        name = "LSTRSegmentationLoss",
        num_classes = 11,
        ignore_index = -100
    )
)