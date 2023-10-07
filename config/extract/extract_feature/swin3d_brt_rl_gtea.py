'''
Author       : Thyssen Wen
Date         : 2023-04-11 13:17:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-11 20:51:14
Description  : file content
FilePath     : /SVTAS/config/extract/extract_feature/swin3d_brt_rl_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 64
clip_seg_num = 64
output_dir_name = 'extract_cls_features'

MODEL = dict(
    architecture = "StreamSegmentation3DWithBackbone",
    backbone = dict(
        name = "SwinTransformer3D",
        # pretrained = "./data/checkpoint/swin_base_patch244_window877_kinetics600_22k.pth",
        # pretrained2d = False,
        patch_size = [2, 4, 4],
        embed_dim = 128,
        depths = [2, 2, 18, 2],
        num_heads = [4, 8, 16, 32],
        window_size = [8,7,7],
        mlp_ratio = 4.,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.2,
        patch_norm = True,
        # graddrop_config={"gd_downsample": 1, "with_gd": [[1, 1], [1, 1], [1] * 14 + [0] * 4, [0, 0]]}
    ),
    neck = dict(
        name = "TaskFusionPoolNeck",
        num_classes=11,
        in_channels = 1024,
        clip_seg_num = clip_seg_num // 2,
        need_pool = True
    ),
    head = dict(
        name = "BRTSegmentationHead",
        num_head=1,
        state_len=512,
        causal=False,
        num_decoders=3,
        encoder_num_layers=8,
        decoder_num_layers=8,
        num_f_maps=128,
        dropout=0.5,
        input_dim=1024,
        num_classes=11,
        channel_masking_rate=0.2,
        sample_rate=sample_rate * 2,
        out_feature=True
    ),
    loss = None
)

PRETRAINED = "./output/final_RGB_gtea_mcepoch80_SwinTransformer3D_BRT_64x2_gtea_split4_best.pt"

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 1,
    num_workers = 2,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./output/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    )
)

DATASETPIPLINE = dict(
    name = "BaseDatasetPipline",
    decode = dict(
        name = "VideoDecoder",
        backend = dict(name="DecordContainer")
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
        clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
        sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
        sample_add_key_pair={"frames":"imgs"},
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "VideoTransform",
        transform_dict = dict(
            imgs = [
            dict(ResizeImproved = dict(size = 256)),
            dict(CenterCrop = dict(size = 224)),
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(Normalize = dict(
                mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
            ))
        ])
    )
)