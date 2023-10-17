'''
Author       : Thyssen Wen
Date         : 2023-10-09 18:38:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 20:34:26
Description  : file content
FilePath     : /SVTAS/config/svtas/diffact/breakfast/stream_diffact_breakfast.py
'''
_base_ = [
    '../../../_base_/dataloader/collater/batch_compose.py',
    '../../../_base_/engine/standaline_engine.py',
    '../../../_base_/logger/python_logger.py',
]

split = 1
num_classes = 48
ignore_index = -100
epochs = 800
batch_size = 1
in_channels = 2048
sample_rate = 8
clip_seg_num = 128
sliding_window = clip_seg_num * sample_rate
sigma = 1
model_name = "Stream_Diffact_feature_breakfast_split" + str(split)

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
                name = "FeatureSegmentation",
                architecture_type='1d',
                head = dict(
                    name = "DiffsusionActionSegmentationEncoderModel",
                    input_dim = 2048,
                    num_classes = num_classes,
                    sample_rate = sample_rate,
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
            sample_rate = sample_rate,
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
        name = "StreamSegmentationLoss",
        backbone_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = 1,
            smooth_weight = 0.0,
            ignore_index = ignore_index
        ),
        head_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate,
            smooth_weight = 0.0,
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
        name = "MultiStepLR",
        step_size = [epochs],
        gamma = 0.1
    )
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    
    batch_size = batch_size,
    num_workers = 2
)

COLLATE = dict(
    train=dict(
        to_tensor_keys = ["feature", "flows", "res", "labels", "masks", "precise_sliding_num", "labels_onehot", "boundary_prob"]
    ),
    test=dict(
        to_tensor_keys = ["feature", "flows", "res", "labels", "masks", "precise_sliding_num", "labels_onehot", "boundary_prob"]
    )
)

DATASET = dict(
    train = dict(
        name = "FeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle",
        feature_path = "./data/breakfast/features",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = True,
        sliding_window = sliding_window
    ),
    test = dict(
        name = "FeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle",
        feature_path = "./data/breakfast/features",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = False,
        sliding_window = sliding_window
    )
)

METRIC = dict(
    TAS = dict(
        name = "TASegmentationMetric",
        overlap = [.1, .25, .5],
        actions_map_file_path = "./data/breakfast/mapping.txt",
        file_output = False,
        score_output = False),
    # TAP = dict(
    #     name = "TAProposalMetric",
    #     actions_map_file_path = "./data/breakfast/mapping.txt",
    #     max_proposal=100,),
    # TAL = dict(
    #     name = "TALocalizationMetric",
    #     actions_map_file_path = "./data/breakfast/mapping.txt",
    #     show_ovberlaps=[0.5, 0.75],),
    # SVTAS = dict(
    #     name = "SVTASegmentationMetric",
    #     overlap = [.1, .25, .5],
    #     segment_windows_size = 64,
    #     actions_map_file_path = "./data/breakfast/mapping.txt",
    #     file_output = False,
    #     score_output = False),
)

DATASETPIPLINE = dict(
    train = dict(
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
            is_train = True,
            sample_rate_dict={"feature":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"feature":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"feature":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"feature"},
            ignore_index=ignore_index,
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_results_list = [
                dict(DropResultsByKeyName = dict(drop_keys_list=[
                    "filename", "raw_labels", "sample_sliding_idx", "format", "frames", "frames_len", "feature_len", "video_len"
                ])),
                dict(RenameResultTransform = dict(rename_pair_dict=dict(
                    video_name = "vid_list"
                )))
            ],
            transform_key_dict = dict(
                feature = [dict(XToTensor = None)],
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
                ),
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
            sample_rate_dict={"feature":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"feature":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"feature":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"feature"},
            ignore_index=ignore_index,
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_results_list = [
                dict(DropResultsByKeyName = dict(drop_keys_list=[
                    "filename", "raw_labels", "sample_sliding_idx", "format", "frames", "frames_len", "feature_len", "video_len"
                ])),
                dict(RenameResultTransform = dict(rename_pair_dict=dict(
                    video_name = "vid_list"
                )))
            ],
            transform_key_dict = dict(
                feature = [dict(XToTensor = None)],
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
                ),
                masks = dict(
                    masks = dict(name = 'direct_transform',
                                 transforms_op_list = [
                                     dict(NumpyDataTypeTransform = dict(dtype = "float32"))])
                )
            )
        )
    )
)