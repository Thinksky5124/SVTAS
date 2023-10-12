'''
Author       : Thyssen Wen
Date         : 2023-10-09 18:38:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-12 10:51:36
Description  : file content
FilePath     : /SVTAS/config/svtas/diffact/raw_diffact_gtea.py
'''
_base_ = [
    '../../_base_/dataloader/collater/stream_compose.py',
    '../../_base_/engine/train_engine.py',
    '../../_base_/logger/python_logger.py',
]

split = 1
num_classes = 11
ignore_index = -100
epochs = 80
batch_size = 1
in_channels = 2048
clip_seg_num_list = [64, 128, 256]
sample_rate_list = [2]
sample_rate = 2
clip_seg_num = 64
sliding_window = clip_seg_num * sample_rate
model_name = "Dynamic_Stream_Diffact_feature_gtea_split" + str(split)

ENGINE = dict(
    name = "StandaloneEngine",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = epochs,
        batch_size = batch_size,
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
        name = "FeatureSegmentation",
        architecture_type ='1d',
        backbone = None,
        neck = None,
        head = dict(
            name = "MultiStageModel",
            num_stages = 4,
            num_layers = 10,
            num_f_maps = 64,
            dim = in_channels,
            num_classes = num_classes,
            sample_rate = 2
        )
    ),
    post_processing = dict(
        name = "StreamScorePostProcessing",
        ignore_index = ignore_index
    ),
    criterion = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = 2,
        smooth_weight = 0.0,
        ignore_index = ignore_index
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
        name = "MultiStepLR",
        step_size = [epochs],
        gamma = 0.1
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
        name = "RawFrameDynamicStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        dynamic_stream_generator=dict(
            name = "MultiEpochStageDynamicStreamGenerator",
            multi_epoch_list = [2, 5],
            strategy_list = [
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = clip_seg_num_list,
                     sample_rate_list = sample_rate_list),
                dict(name = "RandomDynamicStreamGenerator",
                     clip_seg_num_range_list = [64, 128],
                     sample_rate_range_list = [2, 2]),
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = clip_seg_num_list,
                     sample_rate_list = sample_rate_list),
            ]
        )
    ),
    test = dict(
        name = "FeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        feature_path = "./data/gtea/raw_features",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    )
)

METRIC = dict(
    TAS = dict(
        name = "TASegmentationMetric",
        overlap = [.1, .25, .5],
        actions_map_file_path = "./data/gtea/mapping.txt",
        file_output = False,
        score_output = False),
    # TAP = dict(
    #     name = "TAProposalMetric",
    #     actions_map_file_path = "./data/gtea/mapping.txt",
    #     max_proposal=100,),
    # TAL = dict(
    #     name = "TALocalizationMetric",
    #     actions_map_file_path = "./data/gtea/mapping.txt",
    #     show_ovberlaps=[0.5, 0.75],),
    # SVTAS = dict(
    #     name = "SVTASegmentationMetric",
    #     overlap = [.1, .25, .5],
    #     segment_windows_size = 64,
    #     actions_map_file_path = "./data/gtea/mapping.txt",
    #     file_output = False,
    #     score_output = False),
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
            name='FeatureDecoder',
            backend=dict(
                    name='NPYContainer',
                    is_transpose=False,
                    temporal_dim=-1,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 )
        ),
        sample = dict(
            name = "FeatureStreamSampler",
            is_train = False,
            sample_rate_dict={"feature":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"feature":clip_seg_num, "labels":clip_seg_num},
            sliding_window_dict={"feature":sliding_window, "labels":sliding_window},
            sample_add_key_pair={"frames":"feature"},
            feature_dim_dict={"feature":in_channels},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    )
)