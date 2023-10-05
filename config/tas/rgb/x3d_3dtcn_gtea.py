'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:07:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-02 22:42:40
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/x3d_3dtcn_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_video.py'
]
split = 1
num_classes = 11
sample_rate = 1
clip_seg_num = 128
ignore_index = -100
batch_size = 1
epochs = 50
model_name = "X3D_M_3DTCN_gtea_split" + str(split)

MODEL = dict(
    architecture = "Segmentation3D",
    backbone = dict(
        name = "X3D",
        # pretrained="data/checkpoint/x3d_m.pyth",
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
    neck = None,
    head = dict(
        name = "TCN3DHead",
        seg_in_channels = 192,
        num_layers = 6,
        num_f_maps = 64,
        num_classes = num_classes,
        sample_rate = sample_rate,
        num_stages = 1
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
        name = "BaseDatasetPipline",
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
            name = "VideoTransform",
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
        name = "BaseDatasetPipline",
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
            name = "VideoTransform",
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