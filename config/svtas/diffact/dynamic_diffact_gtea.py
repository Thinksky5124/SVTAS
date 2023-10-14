'''
Author       : Thyssen Wen
Date         : 2023-10-09 18:38:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 21:48:40
Description  : file content
FilePath     : /SVTAS/config/svtas/diffact/dynamic_diffact_gtea.py
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
sigma = 1
in_channels = 2048
clip_seg_num_list = [64, 128, 256]
sample_rate_list = [2]
sample_rate = 2
clip_seg_num = 64
sliding_window = clip_seg_num * sample_rate
model_name = "Dynamic_Stream_Diffact_gtea_split" + str(split)

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
        name = "TemporalActionSegmentationDiffusionModel",
        vae = dict(
            name = "TemporalActionSegmentationVariationalAutoEncoder",
            encoder = dict(
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
                    name = "DiffsusionActionSegmentationEncoderModel",
                    input_dim = 1024,
                    num_classes = num_classes,
                    sample_rate = sample_rate * 2,
                    num_layers = 10,
                    num_f_maps = 64,
                    kernel_size = 5,
                    attn_dropout_rate = 0.5,
                    channel_dropout_rate = 0.5,
                    temporal_dropout_rate = 0.5,
                    feature_layer_indices = [5, 7, 9]
                )
            ),
            decoder = None
        ),
        unet = dict(
            name = "DiffsusionActionSegmentationConditionUnet",
            input_dim = 192,
            num_classes = num_classes,
            ignore_index = ignore_index,
            sample_rate = sample_rate * 2,
            num_layers = 8,
            num_f_maps = 24,
            time_emb_dim = 512,
            kernel_size = 5,
            attn_dropout_rate = 0.1
        ),
        scheduler = dict(
            name = "DiffsusionActionSegmentationScheduler",
            num_train_timesteps = 1000,
            num_inference_steps = 25,
            ddim_sampling_eta = 1.0,
            snr_scale = 0.5,
            timestep_spacing = 'linspace',
            infer_region_seed = 8
        )
    ),
    post_processing = dict(
        name = "StreamScorePostProcessing",
        ignore_index = ignore_index
    ),
    criterion = dict(
        name = "DiffusionStreamSegmentationLoss",
        backbone_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate,
            smooth_weight = 0.0,
            ignore_index = ignore_index
        ),
        head_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate,
            smooth_weight = 0.0,
            ignore_index = ignore_index
        ),
        vae_backbone_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate * 2,
            smooth_weight = 0.0,
            ignore_index = -100
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
        name = "DiffusionRawFrameDynamicStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        dynamic_stream_generator=dict(
            name = "MultiEpochStageDynamicStreamGenerator",
            multi_epoch_list = [40, 70],
            strategy_list = [
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [64],
                     sample_rate_list = sample_rate_list),
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [64, 32],
                     sample_rate_list = sample_rate_list),
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [16, 32, 64],
                     sample_rate_list = sample_rate_list),
            ]
        )
    ),
    test = dict(
        name = "DiffusionRawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    )
)

COLLATE = dict(
    train=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num", "labels_onehot", "boundary_prob"]
    ),
    test=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num", "labels_onehot", "boundary_prob"]
    ),
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
            name = "VideoDynamicStreamSampler",
            is_train = True,
            sample_rate_name_dict={"imgs":'sample_rate', "labels":'sample_rate'},
            clip_seg_num_name_dict={"imgs": 'clip_seg_num', "labels": 'clip_seg_num'},
            ignore_index=ignore_index,
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
                ))],
                labels = dict(
                    labels_onehot = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(LabelsToOneHot = dict(
                                num_classes = num_classes,
                                ignore_index = ignore_index
                            ))
                        ]
                    ),
                    boundary_prob = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(SegmentationLabelsToBoundaryProbability = dict(
                                sigma = sigma,
                                need_norm = True
                            ))
                        ]
                    )
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
            transform_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 224)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                    ))],
                labels = dict(
                    labels_onehot = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(LabelsToOneHot = dict(
                                num_classes = num_classes,
                                ignore_index = ignore_index
                            ))
                        ]
                    ),
                    boundary_prob = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(SegmentationLabelsToBoundaryProbability = dict(
                                sigma = sigma,
                                need_norm = True
                            ))
                        ]
                    )
                )
            )
        )
    )
)