'''
Author       : Thyssen Wen
Date         : 2022-10-25 21:28:26
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 19:26:27
Description  : SwinTransformer Config
FilePath     : /SVTAS/config/extract/extract_feature/swin_transformer_rgb_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py'
]

sample_rate = 2
ignore_index = -100
sliding_window = 1
clip_seg_num = 32

MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "SwinTransformer3D",
        pretrained = "./data/swin_tiny_patch244_window877_kinetics400_1k.pth",
        pretrained2d = False,
        patch_size = [2, 4, 4],
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = [8,7,7],
        mlp_ratio = 4.,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.2,
        patch_norm = True
    ),
    neck = None,
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 768,
        input_seg_num = 16,
        output_seg_num = 1,
        sample_rate = sample_rate,
        pool_space = True,
        in_format = "N*T,C,H,W",
        out_format = "NCT"
    ),
    loss = None
)

PRETRAINED = None

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 4,
    num_workers = 2,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window,
        clip_seg_num = clip_seg_num,
        sample_rate = sample_rate
    )
)

PIPELINE = dict(
    name = "BasePipline",
    decode = dict(
        name = "VideoDecoder",
        backend = "decord"
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        sliding_window = sliding_window,
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "VideoStreamTransform",
        transform_list = [
            dict(ResizeImproved = dict(size = 256)),
            dict(CenterCrop = dict(size = 224)),
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(Normalize = dict(
                mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
            ))
        ]
    )
)