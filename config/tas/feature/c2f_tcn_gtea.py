'''
Author       : Thyssen Wen
Date         : 2023-02-24 15:31:44
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-24 15:41:29
Description  : file content
FilePath     : /SVTAS/config/tas/feature/c2f_tcn_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/c2f_tcn.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_feature.py'
]

split = 2
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 500
batch_size = 1
model_name = "C2F_TCN_gtea_split" + str(split)

MODEL = dict(
    head = dict(
        name = "C2F_TCN",
        n_channels = 2048,
        num_classes = num_classes,
        ensem_weights = [1, 1, 1, 1, 0, 0],
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
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        # feature_path = "./data/gtea/raw_features"
        # flow_feature_path = "./data/gtea/flow_features"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        # feature_path = "./data/gtea/raw_features"
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