'''
Author       : Thyssen Wen
Date         : 2023-02-25 14:50:06
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-25 14:50:33
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/lstm.py
'''
MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "LSTMSegmentationHead",
        in_channels = 512,
        num_classes = 11,
        sample_rate = 1,
        hidden_channels=1024,
        num_layers=3,
        batch_first=True,
        dropout=0.5,
        bidirectional=False,
        is_memory_sliding=False
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 1,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)