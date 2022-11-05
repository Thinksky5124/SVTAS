'''
Author       : Thyssen Wen
Date         : 2022-11-03 20:04:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 20:55:51
Description  : file content
FilePath     : /SVTAS/config/svtas/feature/lstr_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/lstr.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 50
long_memory_num_samples = 512
work_memory_num_samples = 32
clip_seg_num = long_memory_num_samples + work_memory_num_samples
sliding_window = work_memory_num_samples
visual_size = 1024
motion_size = 1024
model_name = "Stream_LSTR_gtea_split" + str(split)

MODEL = dict(
    head = dict(
        name = "LSTR",
        modality='twostream',
        visual_size=visual_size,
        motion_size=motion_size,
        linear_enabled=True,
        linear_out_features=1024,
        long_memory_num_samples=long_memory_num_samples,
        work_memory_num_samples=work_memory_num_samples,
        num_heads=16,
        dim_feedforward=1024,
        dropout=0.2,
        activation='relu',
        num_classes=num_classes,
        enc_module=[
                [16, 1, True], [32, 2, True]
                ],
        dec_module=[-1, 2, True],
        sample_rate=sample_rate
    ),
    loss = dict(
        name = "LSTRSegmentationLoss",
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
        feature_path = "./data/gtea/raw_features",
        need_precise_grad_accumulate = False,
        sliding_window = sliding_window
        # flow_feature_path = "./data/gtea/flow_features"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        feature_path = "./data/gtea/raw_features",
        need_precise_grad_accumulate = False,
        sliding_window = sliding_window
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
            name = "FeatureStreamSampler",
            is_train = True,
            sample_rate = sample_rate,
            sample_mode = "uniform",
            sliding_window = sliding_window,
            clip_seg_num = clip_seg_num,
            feature_dim = visual_size + motion_size
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
            feature_dim = visual_size + motion_size
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_list = [
                dict(XToTensor = None)
            ]
        )
    )
)
