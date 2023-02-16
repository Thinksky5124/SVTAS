'''
Author       : Thyssen Wen
Date         : 2022-11-05 15:00:40
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-15 10:33:33
Description  : file content
FilePath     : /SVTAS/config/tas/feature/tasegformer_breakfast.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/breakfast/breakfast_feature.py'
]

split = 1
num_classes = 48
sample_rate = 1
ignore_index = -100
epochs = 50
batch_size = 2
log_interval = 100
model_name = "TASegformer_breakfast_split" + str(split)

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "TASegFormer",
        in_channels=2048,
        num_decoders=1,
        decoder_num_layers=10,
        encoder_num_layers=10,
        input_dropout_rate=0.3,
        embed_dim=64,
        dropout=0.5,
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

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    num_workers = 4,
    train = dict(
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle",
        # feature_path = "./data/breakfast/raw_features"
        # flow_feature_path = "./data/breakfast/flow_features"
    ),
    test = dict(
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle",
        # feature_path = "./data/breakfast/raw_features"
        # flow_feature_path = "./data/breakfast/flow_features"
    )
)

OPTIMIZER = dict(
    name = "AdamWOptimizer",
    learning_rate = 0.001,
    weight_decay = 0.01,
    betas = (0.9, 0.999)
)

LRSCHEDULER = dict(
    step_size = [epochs]
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