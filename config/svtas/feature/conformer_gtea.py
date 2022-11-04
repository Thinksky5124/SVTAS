'''
Author       : Thyssen Wen
Date         : 2022-11-03 20:04:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 21:12:15
Description  : file content
FilePath     : /SVTAS/config/svtas/feature/conformer_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/conformer.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 50
clip_seg_num = 32
sliding_window = 32
model_name = "Stream_Conformer_gtea_split" + str(split)

MODEL = dict(
    head = dict(
        input_dim = 2048,
        num_encoder_layers = 4,
        input_dropout_p = 0.5,
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
    name = "StreamScorePostProcessing",
    sliding_window = sliding_window,
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
            name = "FeatureStreamSampler",
            is_train = True,
            sample_rate = sample_rate,
            sample_mode = "uniform",
            sliding_window = sliding_window,
            clip_seg_num = clip_seg_num,
            feature_dim = 2048
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
            name = "FeatureStreamSampler",
            is_train = False,
            sample_rate = sample_rate,
            sample_mode = "uniform",
            sliding_window = sliding_window,
            clip_seg_num = clip_seg_num,
            feature_dim = 2048
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_list = [
                dict(XToTensor = None)
            ]
        )
    )
)
