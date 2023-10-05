'''
Author       : Thyssen Wen
Date         : 2022-11-05 15:00:40
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-13 15:41:26
Description  : file content
FilePath     : /SVTAS/config/tas/feature/asformer_ipb_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    # '../../_base_/models/temporal_action_segmentation/asformer.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 50
batch_size = 1
gop_size = 16
model_name = "Asformer_IPB_16_gtea_split" + str(split)

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = dict(
        name = "IPBFusionNeck",
        gop_size=gop_size,
        spatial_expan_mode='bilinear'
    ),
    head = dict(
        name = "ASFormer",
        num_decoders = 2,
        num_layers = 4,
        r1 = 2,
        r2 = 2,
        num_f_maps = 64,
        input_dim = 2048,
        num_classes = num_classes,
        sample_rate = sample_rate,
        channel_masking_rate = 0.5
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
            transform_list = [
                dict(XToTensor = None)
            ]
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
            transform_list = [
                dict(XToTensor = None)
            ]
        )
    )
)