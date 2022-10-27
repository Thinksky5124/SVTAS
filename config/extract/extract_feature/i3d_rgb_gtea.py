'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:53:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 19:21:41
Description  : I3D Extractor Config
FilePath     : /SVTAS/config/extract/extract_feature/i3d_rgb_gtea.py
'''

_base_ = [
    '../../_base_/collater/stream_compose.py', '../../_base_/models/action_recognition/i3d.py',
]

sample_rate = 1
ignore_index = -100
sliding_window = 1
clip_seg_num = 64

MODEL = dict(
    head = dict(
        sample_rate = sample_rate
    )
)

PRETRAINED = None

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    config = dict(
        sliding_window = sliding_window,
        clip_seg_num = clip_seg_num,
        sample_rate = sample_rate
    )
)

PIPELINE = dict(
    name = "BasePipline",
    decode = dict(
        name = "VideoDecoder",
        backend = "decord"
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        sliding_window = sliding_window,
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "VideoStreamTransform",
        transform_list = [
            dict(ResizeImproved = dict(size = 256)),
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(TensorCenterCrop = dict(crop_size = 224)),
            dict(ScaleTo1_1 = None)
        ]
    )
)