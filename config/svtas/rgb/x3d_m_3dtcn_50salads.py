'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:07:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-02 20:38:49
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/x3d_m_3dtcn_50salads.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/50salads/50salads_stream_video.py'
]
split = 1
num_classes = 19
sample_rate = 8
clip_seg_num = 128
ignore_index = -100
batch_size = 1
sliding_window = 512
epochs = 50
model_name = "X3D_M_3DTCN_50salads_split" + str(split)

MODEL = dict(
    architecture = "StreamSegmentation3D",
    backbone = dict(
        name = "X3D",
        pretrained="data/checkpoint/x3d_m.pyth",
        dim_c1=12,
        scale_res2=False,
        depth=50,
        num_groups=1,
        width_per_group=64,
        width_factor=2.0,
        depth_factor=2.2,
        input_channel_num=[3],
        bottleneck_factor=2.25,
        channelwise_3x3x3=True
    ),
    neck = dict(
        name = "AvgPoolNeck",
        num_classes = num_classes,
        in_channels = 192,
        clip_seg_num = clip_seg_num,
        need_pool = False
    ),
    head = dict(
        name = "TCN3DHead",
        seg_in_channels = 192,
        num_layers = 6,
        num_f_maps = 64,
        num_classes = num_classes,
        sample_rate = sample_rate,
        num_stages = 1
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 0.15,
        ignore_index = -100
    )
)

POSTPRECESSING = dict(
    name = "StreamScorePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 4,
    train = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/50salads/splits/test.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    )
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

PIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            name = "VideoDecoder",
            backend = "decord"
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = True,
            sample_mode = 'random',
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            sliding_window = sliding_window
        ),
        transform = dict(
            name = "VideoStreamTransform",
            transform_list = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                ))
            ]
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            name = "VideoDecoder",
            backend = "decord"
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = True,
            sample_mode = 'uniform',
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            sliding_window = sliding_window
        ),
        transform = dict(
            name = "VideoStreamTransform",
            transform_list = [
                dict(ResizeImproved = dict(size = 256)),
                dict(CenterCrop = dict(size = 224)),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                ))
            ]
        )
    )
)