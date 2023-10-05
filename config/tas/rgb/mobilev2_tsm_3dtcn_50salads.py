'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:07:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 21:46:09
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/mobilev2_tsm_3dtcn_50salads.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/50salads/50salads_video.py'
]
split = 1
num_classes = 19
sample_rate = 1
clip_seg_num = 128
ignore_index = -100
batch_size = 1
epochs = 50
model_name = "MobileV2_TSM_MS_TCN_50salads_split" + str(split)

MODEL = dict(
    architecture = "Segmentation2D",
    backbone = dict(
        name = "MobileNetV2TSM",
        # pretrained = "./data/checkpoint/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth",
        clip_seg_num = clip_seg_num,
        shift_div = 8,
        out_indices = (7, )
    ),
    neck = dict(
       name = "PoolNeck",
        in_channels = 1280,
        clip_seg_num = clip_seg_num,
        drop_ratio = 0.5,
        need_pool = True
    ),
    head = dict(
        name = "MultiStageModel",
        num_stages = 3,
        num_layers = 6,
        num_f_maps = 64,
        dim = 1280,
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

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle"
    ),
    test = dict(
        file_path = "./data/50salads/splits/test.split" + str(split) + ".bundle"
    )
)

LRSCHEDULER = dict(
    step_size = [epochs]
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
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
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
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                ))
            ]
        )
    )
)