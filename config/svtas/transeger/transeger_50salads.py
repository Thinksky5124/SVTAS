'''
Author       : Thyssen Wen50salads
Date         : 2023-10-07 19:06:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 20:25:06
Description  : file content
FilePath     : /SVTAS/config/svtas/transeger/transeger_50salads.py
'''
_base_ = [
    '../../_base_/dataloader/collater/batch_compose.py',
    '../../_base_/engine/standaline_engine.py',
    '../../_base_/logger/python_logger.py',
    '../../_base_/dataloader/dataset/50salads/50salads_stream_video.py'
]

num_classes = 19
sample_rate = 4
clip_seg_num = 64
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 50

model_name = "Transeger_"+str(clip_seg_num)+"x"+str(sample_rate)+"_50salads_split" + str(split)

ENGINE = dict(
    name = "StandaloneEngine",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = epochs,
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
        name = "Transeger",
        image_backbone = dict(
            name = "Recognition2D",
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
            name = "Encoder2Decoder",
            encoder = dict(
                name = "LearnerPromptTextEncoder",
                actions_map_file_path = "./data/50salads/mapping.txt",
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

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    
    batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    train = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
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
            transform_results_list = [
                dict(DropResultsByKeyName = dict(drop_keys_list=[
                    "filename", "raw_labels", "sample_sliding_idx", "format", "frames", "frames_len", "feature_len", "video_len"
                ])),
                dict(RenameResultTransform = dict(rename_pair_dict=dict(
                    video_name = "vid_list"
                )))
            ],
            transform_key_dict = dict(
                imgs = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                ))],
                masks = dict(
                    masks = dict(name = 'direct_transform',
                                 transforms_op_list = [
                                     dict(NumpyDataTypeTransform = dict(dtype = "float32"))])
                )
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
            transform_results_list = [
                dict(DropResultsByKeyName = dict(drop_keys_list=[
                    "filename", "raw_labels", "sample_sliding_idx", "format", "frames", "frames_len", "feature_len", "video_len"
                ])),
                dict(RenameResultTransform = dict(rename_pair_dict=dict(
                    video_name = "vid_list"
                )))
            ],
            transform_key_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 224)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                        std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                    ))],
                masks = dict(
                    masks = dict(name = 'direct_transform',
                                 transforms_op_list = [
                                     dict(NumpyDataTypeTransform = dict(dtype = "float32"))])
                )
            )
        )
    )
)