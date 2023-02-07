'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:24:30
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-07 13:43:22
Description  : file content
FilePath     : /SVTAS/config/tas/feature/linformer_gtea.py
'''

_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/models/temporal_action_segmentation/linformer.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_feature.py', '../../_base_/schedules/lr/liner_step_50e.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
model_name = "Linformer_gtea_split" + str(split)
batch_size = 1
epochs = 50
log_interval = 8

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name ="LinformerHead",
        num_decoders = 3,
        num_layers = 10,
        num_f_maps = 64,
        input_dim = 2048,
        channel_masking_rate = 0.5,
        num_classes=num_classes,
        sample_rate=sample_rate
    ),
    loss = dict(
        name = "DiceSegmentationLoss",
        smooth_weight = 0.0,
        num_classes = num_classes,
        sample_rate = sample_rate,
        ignore_index = ignore_index
    )
)

POSTPRECESSING = dict(
    name = "ScorePostProcessing",
    ignore_index = ignore_index
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

OPTIMIZER = dict(
    name = "AdamWOptimizer",
    learning_rate = 0.0005,
    weight_decay = 0.01,
    betas = (0.9, 0.999)
)

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        # feature_path = "./data/gtea/raw_features",
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        # feature_path = "./data/gtea/raw_features",
    )
)

METRIC = dict(
    TAS = dict(
        file_output = True,
        score_output = False)
)

PIPELINE = dict(
    train = dict(
        name = "BasePipline",
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
            name = "FeatureSampler",
            is_train = True,
            sample_rate_dict={ "feature": sample_rate,"labels": sample_rate },
            sample_add_key_pair={ "frames": "feature" },
            sample_mode = "uniform",
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    ),
    test = dict(
        name = "BasePipline",
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
            name = "FeatureSampler",
            is_train = False,
            sample_rate_dict={ "feature": sample_rate,"labels": sample_rate },
            sample_add_key_pair={ "frames": "feature" },
            sample_mode = "uniform",
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    )
)
