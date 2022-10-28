'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:24:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:31:00
Description  : file content
FilePath     : /SVTAS/config/tas/feature/ms_tcn/ms_tcn_gtea.py
'''

_base_ = [
    '../../../_base_/schedules/adam_100e.py', '../../../_base_/models/temporal_action_segmentation/ms_tcn.py',
    '../../../_base_/default_runtime.py', '../../../_base_/collater/batch_compose.py',
    '../../../_base_/dataset/gtea/gtea_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
model_name = "MSTCN_gtea_split1"

MODEL = dict(
    head = dict(
        dim = 1536,
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
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        flow_feature_path = "./data/gtea/flow_features"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        flow_feature_path = "./data/gtea/flow_features"
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
                dict(FeatureToTensor = None)
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
                dict(FeatureToTensor = None)
            ]
        )
    )
)
