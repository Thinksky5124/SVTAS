'''
Author       : Thyssen Wen
Date         : 2022-10-28 16:07:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 14:36:50
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/mobilev2_tsm_3dtcn_50salads.py
'''
_base_ = [
    '../../_base_/schedules/adam_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/50salads/50salads_video.py'
]
split = 1
num_classes = 19
sample_rate = 1
clip_seg_num = 128
ignore_index = -100
batch_size = 1
model_name = "SLViT_3DTCN_50salads_split" + str(split)

MODEL = dict(
    architecture = "Segmentation2D",
    backbone = dict(
        name = "SLViT",
        image_size = 224,
        patch_size = 32,
        depth = 4,
        heads = 12,
        mlp_dim = 1024,
        dropout = 0.5,
        emb_dropout = 0.5
    ),
    neck = dict(
        name = "AvgPoolNeck",
        num_classes = num_classes,
        in_channels = 1024,
        clip_seg_num = clip_seg_num,
        drop_ratio = 0.5,
        need_pool = False
    ),
    head = dict(
        name = "TCN3DHead",
        seg_in_channels = 1024,
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
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle"
    ),
    test = dict(
        file_path = "./data/50salads/splits/test.split" + str(split) + ".bundle"
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
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                ))
            ]
        )
    )
)