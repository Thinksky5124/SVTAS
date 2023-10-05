'''
Author       : Thyssen Wen
Date         : 2023-02-08 11:38:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-03-20 18:06:33
Description  : file content
FilePath     : /SVTAS/config/tas/feature/asrf_breakfast.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/asrf.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/breakfast/breakfast_feature.py'
]

split = 1
num_classes = 48
sample_rate = 1
ignore_index = -100
epochs = 50
in_channels = 2048
log_interval = 100
model_name = "ASRF_breakfast_split" + str(split)

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone=None,
    neck=None,
    head = dict(
        name = "ActionSegmentRefinementFramework",
        in_channel = in_channels,
        num_features = 64,
        num_stages = 4,
        num_layers = 10,
        num_classes = num_classes,
        sample_rate = sample_rate
    ),
    loss = dict(
        name = "ASRFLoss",
        class_weight = [1.0]*48,
        pos_weight = [1.0],
        # class_weight = [0.156247, 0.78171575, 0.28677654, 4.926676, 2.0104852, 1.0036047, 1.944697,
        #                 3.774724, 6.955852, 10.38049, 1.2343992, 0.23753178, 0.5512463, 0.12770891,
        #                 0.56181335, 0.538746, 11.917327, 0.34326395, 5.027829, 1.7655491, 0.29985616,
        #                 1.0759093, 2.7336922, 14.338625, 0.69723165, 0.9964213, 0.7521232, 0.24145696,
        #                 0.64888424, 0.09441632, 1.4241172, 7.372144, 0.13262297, 0.9556162, 0.24871513,
        #                 6.752492, 0.88652873, 0.41125828, 5.716496, 0.62820673, 7.748761, 41.90722,
        #                 1.2652122, 1.7691605, 0.19078597, 1.0788504, 0.97267425, 38.899525],
        # pos_weight = [308.0730096087852],
        num_classes = num_classes,
        sample_rate = sample_rate,
        ignore_index = -100
    )
)

POSTPRECESSING = dict(
    name = "ScorePostProcessingWithRefine",
    ignore_index = ignore_index,
    refine_method_cfg = dict(
        name = "ASRFRefineMethod",
        refinement_method="refinement_with_boundary",
        boundary_threshold=0.5,
        theta_t=15,
        kernel_size=15
    )
)

DATASET = dict(
    train = dict(
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle",
        # flow_feature_path = "./data/breakfast/flow_features"
    ),
    test = dict(
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle",
        # flow_feature_path = "./data/breakfast/flow_features"
    )
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

PIPELINE = dict(
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