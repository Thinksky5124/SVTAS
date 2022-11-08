'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:24:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-07 15:33:05
Description  : file content
FilePath     : /SVTAS/config/tas/feature/ms_tcn_ipb_50salads.py
'''

_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    # '../../_base_/models/temporal_action_segmentation/ms_tcn.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/50salads/50salads_feature.py'
]

split = 1
num_classes = 19
sample_rate = 1
ignore_index = -100
epochs = 50
model_name = "MSTCN_IPB_50salads_split" + str(split)

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = dict(
        name = "IPBFusionNeck",
        gop_size=30,
        spatial_expan_mode='bilinear'
    ),
    head = dict(
        name = "MultiStageModel",
        num_stages = 4,
        num_layers = 10,
        num_f_maps = 64,
        dim = 2048,
        num_classes = num_classes,
        sample_rate = sample_rate
    ),
    loss = dict(
        name = "SegmentationLoss",
        smooth_weight = 0.15,
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
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
        # flow_feature_path = "./data/50salads/flow_features"
    ),
    test = dict(
        file_path = "./data/50salads/splits/test.split" + str(split) + ".bundle",
        # flow_feature_path = "./data/50salads/flow_features"
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
