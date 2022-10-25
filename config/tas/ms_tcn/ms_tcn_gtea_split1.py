'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:24:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-25 18:20:34
Description  : file content
FilePath     : /SVTAS/config/tas/ms_tcn/ms_tcn_gtea_split1.py
'''

_base_ = [
    '../../_base_/schedules/adam_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py'
]

num_classes = 11
sample_rate = 1
ignore_index = -100

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "MultiStageModel",
        num_stages = 4,
        num_layers = 10,
        num_f_maps = 64,
        dim = 768,
        num_classes = num_classes,
        sample_rate = sample_rate
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 0.15,
        ignore_index = ignore_index
    )
)

POSTPRECESSING = dict(
    name = "ScorePostProcessing",
    num_classes = num_classes,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 2,
    video_batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        feature_path = "./data/gtea/features",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        sample_rate = sample_rate
    ),
    test = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        feature_path = "./data/gtea/features",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        sample_rate = sample_rate
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

METRIC = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/gtea/mapping.txt",
    file_output = False,
    score_output = False
)

model_name = "MSTCN_gtea_split1"