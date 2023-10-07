'''
Author       : Thyssen Wen
Date         : 2022-10-28 11:00:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 19:22:10
Description  : Transeger
FilePath     : /SVTAS/config/models/transeger/transeger_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py',
    '../../_base_/engine/train_engine.py',
    '../../_base_/logger/python_logger.py',
    '../../_base_/dataset/gtea/gtea_stream_video.py'
]

num_classes = 11
sample_rate = 4
clip_seg_num = 64
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 50

model_name = "Transeger_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

ENGINE = dict(
    name = "BaseImplementEngine",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = 50,
        batch_size = 1,
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
        )
    ),
    post_processing = dict(
        name = "StreamScorePostProcessing",
        sliding_window = sliding_window,
        ignore_index = ignore_index
    ),
    criterion = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 0.5,
        ignore_index = ignore_index
    ),
    optimizer = dict(
        name = "AdamWOptimizer",
        learning_rate = 0.0005,
        weight_decay = 1e-4,
        betas = (0.9, 0.999),
        finetuning_scale_factor=0.5,
        no_decay_key = [],
        finetuning_key = [],
        freeze_key = [],
    ),
    lr_scheduler = dict(
        name = "MultiStepLR",
        step_size = [epochs],
        gamma = 0.1,
    )
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
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
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
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
                        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                    ))]
            )
        )
    )
)