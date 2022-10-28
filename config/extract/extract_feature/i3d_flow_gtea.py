'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:53:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 10:48:46
Description  : I3D Extractor Config
FilePath     : /SVTAS/config/extract/extract_feature/i3d_flow_gtea.py
'''

_base_ = [
    '../../_base_/collater/stream_compose.py', '../../_base_/models/action_recognition/i3d.py',
    '../../_base_/dataset/gtea_video.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 1
clip_seg_num = 64

MODEL = dict(
    backbone = dict(
        pretrained = "./data/i3d_flow.pt",
        in_channels = 2
    ),
    head = dict(
        sample_rate = sample_rate,
    ),
    loss = None
)

PRETRAINED = None

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 4,
    num_workers = 2,
    config = dict(
        videos_path = "./data/gtea/flow",
        sliding_window = sliding_window
    )
)

PIPELINE = dict(
    name = "BasePipline",
    decode = dict(
        name = "FlowVideoDecoder",
        backend = "numpy"
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        sliding_window = sliding_window,
        sample_mode = "uniform",
        channel_mode = "XY"
    ),
    transform = dict(
        name = "VideoStreamTransform",
        transform_list = [
            dict(FeatureToTensor = None),
            dict(ToFloat = None),
            dict(TensorPermute = dict(permute_list = [2, 0, 1])),
            dict(TensorImageResize = dict(size = 256)),
            dict(TensorCenterCrop = dict(crop_size = 224)),
            dict(ScaleTo1_1 = None)
        ]
    )
)