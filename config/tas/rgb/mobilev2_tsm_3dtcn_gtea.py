'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:07:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 20:18:41
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/mobilev2_tsm_3dtcn_gtea.py
'''
_base_ = [
    '../../_base_/schedules/adam_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_video.py'
]
split = 1
num_classes = 11
sample_rate = 1
clip_seg_num = 128
ignore_index = -100
batch_size = 1
model_name = "MobileV2_TSM_3DTCN_gtea_split" + str(split)

MODEL = dict(
    architecture = "Segmentation2D",
    backbone = dict(
        name = "MobileNetV2TSM",
        pretrained = "./data/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth",
        clip_seg_num = clip_seg_num,
        shift_div = 8,
        out_indices = (7, )
    ),
    neck = dict(
        name = "AvgPoolNeck",
        num_classes = num_classes,
        in_channels = 1280,
        clip_seg_num = clip_seg_num,
        drop_ratio = 0.5,
        need_pool = False
    ),
    head = dict(
        name = "TCN3DHead",
        seg_in_channels = 1280,
        num_layers = 4,
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
            name = "VideoDecoder",
            backend = "decord"
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
            name = "VideoDecoder",
            backend = "decord"
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