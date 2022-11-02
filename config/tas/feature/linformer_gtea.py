'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:24:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-01 21:33:45
Description  : file content
FilePath     : /SVTAS/config/tas/feature/linformer_gtea.py
'''

_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/models/temporal_action_segmentation/linformer.py',
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

MODEL = dict(
    head = dict(
        input_dim = 2048,
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

LRSCHEDULER = dict(
    step_size = [epochs]
)

OPTIMIZER = dict(
    learning_rate = 0.0005 * batch_size,
    weight_decay = 1e-4,
    betas = (0.9, 0.999)
)

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        feature_path = "./data/gtea/raw_features",
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        feature_path = "./data/gtea/raw_features",
    )
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
