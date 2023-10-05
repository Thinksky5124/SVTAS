'''
Author       : Thyssen Wen
Date         : 2023-02-08 11:38:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-10 14:17:41
Description  : file content
FilePath     : /SVTAS/config/tas/feature/asrf_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/asrf.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 50
in_channels = 2048
model_name = "ASRF_gtea_split" + str(split)

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
        class_weight = [0.40253314,0.6060787,0.41817436,1.0009843,1.6168522,
                        1.2425169,1.5743035,0.8149039,7.6466165,1.0,0.29321033],
        pos_weight = [33.866594360086765],
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
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        # flow_feature_path = "./data/gtea/flow_features"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        # flow_feature_path = "./data/gtea/flow_features"
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