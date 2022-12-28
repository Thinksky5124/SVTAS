'''
Author       : Thyssen Wen
Date         : 2022-11-05 15:00:40
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-28 17:49:00
Description  : file content
FilePath     : /SVTAS/config/tas/feature/segformer_50salads.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/50salads/50salads_feature.py'
]

split = 1
num_classes = 19
sample_rate = 2
ignore_index = -100
epochs = 50
batch_size = 2
model_name = "Segformer_50salads_split" + str(split)

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "SegFormer",
        in_channels=2048,
        num_decoders=3,
        num_layers=2,
        num_classes=num_classes,
        input_dropout=0.5,
        embed_dim=64,
        num_heads=8,
        dropout=0.5,
        sample_rate=sample_rate,
    ),
    loss = dict(
        name = "DiceSegmentationLoss",
        smooth_weight = 1.0,
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
    train = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
        feature_path = "./data/50salads/features"
        # flow_feature_path = "./data/50salads/flow_features"
    ),
    test = dict(
        file_path = "./data/50salads/splits/test.split" + str(split) + ".bundle",
        feature_path = "./data/50salads/features"
        # flow_feature_path = "./data/50salads/flow_features"
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