'''
Author       : Thyssen Wen
Date         : 2022-11-03 20:59:03
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 20:41:38
Description  : file content
FilePath     : /SVTAS/config/tas/feature/conformer_gtea.py
'''

_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/conformer.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 50
batch_size = 1
model_name = "Conformer_gtea_split" + str(split)

MODEL = dict(
    head = dict(
        input_dim = 2048,
        encoder_dim = 64,
        num_encoder_layers = 8,
        input_dropout_p = 0.3,
        num_attention_heads = 8,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 31,
        half_step_residual = True,
        num_classes = num_classes,
        sample_rate = sample_rate
    ),
    loss = dict(
        num_classes = num_classes,
        sample_rate = sample_rate,
        ignore_index = ignore_index
    )
)

POSTPRECESSING = dict(
    name = "ScorePostProcessing",
    num_classes = num_classes,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        feature_path = "./data/gtea/raw_features"
        # flow_feature_path = "./data/gtea/flow_features"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        feature_path = "./data/gtea/raw_features"
        # flow_feature_path = "./data/gtea/flow_features"
    )
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

PIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            name = "FeatureDecoder",
            backend = "numpy"
        ),
        sample = dict(
            name = "FeatureSampler",
            is_train = True,
            sample_rate = sample_rate,
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_list = [
                dict(XToTensor = None)
            ]
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            name = "FeatureDecoder",
            backend = "numpy"
        ),
        sample = dict(
            name = "FeatureSampler",
            is_train = False,
            sample_rate = sample_rate,
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_list = [
                dict(XToTensor = None)
            ]
        )
    )
)
