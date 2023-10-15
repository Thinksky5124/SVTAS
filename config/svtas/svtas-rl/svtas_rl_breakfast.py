'''
Author       : Thyssen Wen
Date         : 2023-10-07 19:11:47
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 11:09:05
Description  : file content
FilePath     : /SVTAS/config/svtas/svtas-rl/svtas_rl_breakfast.py
'''
_base_ = [
    '../../_base_/dataloader/collater/stream_compose.py',
    '../../_base_/engine/standaline_engine.py',
    '../../_base_/logger/python_logger.py',
    '../../_base_/dataloader/dataset/breakfast/breakfast_stream_video.py'
]

num_classes = 48
sample_rate = 8
clip_seg_num = 128
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 80

model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_breakfast_split" + str(split)

ENGINE = dict(
    name = "StandaloneEngine",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = epochs,
        batch_size = 1,
        logger_iter_interval = 100,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    )
)

MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    grad_accumulate = dict(
        name = "GradAccumulate",
        accumulate_type = "conf"
    ),
    model = dict(
        name = "StreamVideoSegmentation",
        architecture_type ='3d',
        addition_loss_pos = 'with_backbone_loss',
        backbone = dict(
            name = "SwinTransformer3D",
            pretrained = "./data/checkpoint/swin_base_patch244_window877_kinetics600_22k.pth",
            pretrained2d = False,
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
            num_classes=num_classes,
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
            num_classes=num_classes,
            channel_masking_rate=0.2,
            sample_rate=sample_rate * 2
        )
    ),
    post_processing = dict(
        name = "StreamScorePostProcessing",
        sliding_window = sliding_window,
        ignore_index = ignore_index
    ),
    criterion = dict(
        name = "StreamSegmentationLoss",
        backbone_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate * 2,
            smooth_weight = 0.0,
            ignore_index = -100
        ),
        head_loss_cfg = dict(
            name = "RLPGSegmentationLoss",
            num_classes = num_classes,
            smooth_weight = 0.15,
            sample_rate = sample_rate,
            ignore_index = ignore_index
        )
    ),
    optimizer = dict(
        name = "AdamWOptimizer",
        learning_rate = 0.0005,
        weight_decay = 1e-4,
        betas = (0.9, 0.999),
        finetuning_scale_factor=0.02,
        no_decay_key = [],
        finetuning_key = ["backbone."],
        freeze_key = [],
    ),
    lr_scheduler = dict(
        name = "CosineAnnealingLR",
        T_max = epochs,
        eta_min = 0.00001,
    )
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    train = dict(
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/breakfast/Videos_mp4",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle",
        videos_path = "./data/breakfast/Videos_mp4",
        sliding_window = sliding_window,
    )
)

DATASETPIPLINE = dict(
    train = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
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
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                    std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                ))]
            )
        )
    ),
    test = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
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
                        mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                        std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                    ))]
            )
        )
    )
)