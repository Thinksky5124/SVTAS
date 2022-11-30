'''
Author       : Thyssen Wen
Date         : 2022-10-28 11:00:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 11:09:48
Description  : Transeger
FilePath     : /SVTAS/config/_base_/models/temporal_action_segmentation/transeger.py
'''
MODEL = dict(
    architecture = "Transeger",
    image_backbone = dict(
        architecture = "Recognition2D",
        backbone = dict(
            name = "MobileNetV2TSM",
            pretrained = "./data/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth",
            clip_seg_num = 32,
            shift_div = 8,
            out_indices = (7, )
        ),
        neck = dict(
            name = "PoolNeck",
            num_classes = 11,
            in_channels = 1280,
            clip_seg_num = 32,
            drop_ratio = 0.5,
            need_pool = True
        ),
        head = dict(
            name = "IdentityEmbeddingHead",
            in_channels = 1280,
            out_channels = 64,
            sample_rate = 4
        )
    ),
    text_backbone = dict(
        architecture = "Encoder2Decoder",
        encoder = dict(
            name = "LearnerPromptTextEncoder",
            actions_map_file_path = "./data/gtea/mapping.txt",
            embedding_dim = 512,
            sample_rate = 4,
            max_len = 50,
            clip_seg_num = 32,
            encoder_layers_num = 3,
            encoder_heads_num = 8,
            text_embed_dim = 64
        ),
        decoder = None,
        head = None
    ),
    joint = dict(
        name = "TransegerMemoryTCNJointNet",
        num_classes = 11,
        in_channels = 64,
        hidden_channels = 128,
        num_layers = 4,
        sample_rate = 4
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = 11,
        sample_rate = 4,
        smooth_weight = 0.5,
        ignore_index = -100
    )
)
    