'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:59:49
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:09:49
Description  : file content
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/vit_3d_3dtcn.py
'''
MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "ViT",
        image_size = 224,
        patch_size = 16,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
        dropout = 0.1,
        emb_dropout = 0.1,
        # pretrained = 
    ),
    neck = dict(
        name = "AvgPoolNeck",
        num_classes = 11,
        in_channels = 1024,
        clip_seg_num = 8,
        drop_ratio = 0.5,
        need_pool = False
    ),
    head = dict(
        name = "TCN3DHead",
        num_layers = 4,
        num_f_maps = 64,
        seg_in_channels = 1024,
        num_classes = 11,
        sample_rate = 4
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 4,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)