'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:07:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-01 18:23:31
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/timesformer_mstcn_breakfast.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/breakfast/breakfast_video.py'
]
split = 1
num_classes = 48
sample_rate = 1
clip_seg_num = 64
ignore_index = -100
batch_size = 1
epochs = 50
log_interval = 100
model_name = "TimeSformer_MS_TCN_breakfast_split" + str(split)

MODEL = dict(
    architecture = "Segmentation3D",
    backbone = dict(
        name = "TimeSformer",
        pretrained = "./data/checkpoint/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth",
        num_frames = clip_seg_num,
        img_size = 224,
        patch_size = 16,
        embed_dims = 768
    ),
    neck = dict(
        name = "AvgPoolNeck",
        num_classes = num_classes,
        in_channels = 768,
        clip_seg_num = clip_seg_num,
        drop_ratio = 0.5,
        need_pool = True
    ),
    head = dict(
        name = "MultiStageModel",
        num_stages = 1,
        num_layers = 4,
        num_f_maps = 64,
        dim = 768,
        num_classes = num_classes,
        sample_rate = sample_rate
    ),
    aligin_head = dict(
        name = "InterploteAlignHead"
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
    name = "ScorePostProcessing",
    num_classes = num_classes,
    ignore_index = ignore_index
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle"
    ),
    test = dict(
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle"
    )
)

PIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            backend=dict(
                    name='OpenCVContainer')
        ),
        sample = dict(
            name = "VideoSampler",
            is_train = True,
            sample_mode = 'linspace',
            clip_seg_num = clip_seg_num,
            channel_mode="RGB"
        ),
        transform = dict(
            name = "VideoStreamTransform",
            transform_list = [
                dict(OpencvToPIL = dict(in_channel_model = "RGB")),
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                    std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                ))
            ]
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            backend=dict(
                    name='OpenCVContainer')
        ),
        sample = dict(
            name = "VideoSampler",
            is_train = False,
            sample_mode = 'linspace',
            clip_seg_num = clip_seg_num,
            channel_mode = "RGB"
        ),
        transform = dict(
            name = "VideoStreamTransform",
            transform_list = [
                dict(OpencvToPIL = dict(in_channel_model = "RGB")),
                dict(ResizeImproved = dict(size = 256)),
                dict(CenterCrop = dict(size = 224)),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                    std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                ))
            ]
        )
    )
)