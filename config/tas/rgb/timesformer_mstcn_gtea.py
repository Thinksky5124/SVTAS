'''
Author       : Thyssen Wen
Date         : 2022-11-01 16:29:27
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-02 10:23:42
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/timesformer_mstcn_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_video.py'
]
split = 1
num_classes = 11
sample_rate = 1
clip_seg_num = 64
ignore_index = -100
batch_size = 1
epochs = 50
model_name = "TimeSformer_MS_TCN_gtea_split" + str(split)

MODEL = dict(
    architecture = "Segmentation3D",
    backbone = dict(
        name = "TimeSformer",
        pretrained = "./data/checkpoint/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth",
        num_frames = clip_seg_num,
        img_size = 224,
        patch_size = 16,
        embed_dims = 768,
        dropout_ratio = 0.5,
        pool="mean_space"
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
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle"
    )
)

PIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
        ),
        sample = dict(
            name = "VideoSampler",
            is_train = True,
            sample_mode = 'linspace',
            clip_seg_num_dict={"imgs":clip_seg_num, "labels":clip_seg_num},
            sample_add_key_pair={"frames":"imgs"},
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
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ]
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
        ),
        sample = dict(
            name = "VideoSampler",
            is_train = False,
            sample_mode = 'linspace',
            clip_seg_num_dict={"imgs":clip_seg_num, "labels":clip_seg_num},
            sample_add_key_pair={"frames":"imgs"},
        ),
        transform = dict(
            name = "VideoStreamTransform",
            transform_list = [
                dict(ResizeImproved = dict(size = 256)),
                dict(CenterCrop = dict(size = 224)),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ]
        )
    )
)