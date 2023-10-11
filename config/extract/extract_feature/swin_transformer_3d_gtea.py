'''
Author       : Thyssen Wen
Date         : 2022-12-22 16:37:36
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 16:19:05
Description  : file content
FilePath     : /SVTAS/config/extract/extract_feature/swin_transformer_3d_gtea.py
'''
_base_ = [
    '../../_base_/dataloader/collater/stream_compose.py',
    '../../_base_/logger/python_logger.py',
]

sample_rate = 1
batch_size = 4
ignore_index = -100
sliding_window = 128
clip_seg_num = 128

model_name = "SwinTransformer3D_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea"

ENGINE = dict(
    name = "ExtractFeatureEngine",
    out_path = "data/gtea/test/extract_feature",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = 1,
        batch_size = batch_size,
        logger_iter_interval = 10,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    )
)

MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    # pretrained = "",
    model = dict(
        name = "StreamVideoSegmentation",
        architecture_type ='3d',
        backbone = dict(
            name = "SwinTransformer3D",
            pretrained = "./data/checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth",
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
            input_seg_num = clip_seg_num // 2,
            output_seg_num = clip_seg_num,
            sample_rate = sample_rate * 2,
            pool_space = True,
            in_format = "N,C,T,H,W",
            out_format = "NCT"
        )
    ),
    post_processing = dict(
        name = "StreamFeaturePostProcessing",
        sliding_window = sliding_window,
        ignore_index = ignore_index
    ),
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
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