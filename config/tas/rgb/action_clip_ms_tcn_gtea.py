'''
Author       : Thyssen Wen
Date         : 2022-10-30 16:48:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-30 19:26:05
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/action_clip_ms_tcn_gtea.py
'''
_base_ = [
    '../../_base_/schedules/adamw_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/batch_compose.py',
    '../../_base_/dataset/gtea/gtea_video.py'
]
split = 1
num_classes = 11
sample_rate = 1
clip_seg_num = 64
ignore_index = -100
batch_size = 1
model_name = "ActionCLIP_MS_TCN_gtea_split" + str(split)

MODEL = dict(
    architecture = "ActionCLIPSegmentation",
    pretrained = "./data/vit-16-32f.pt",
    is_feature_extract = True,
    image_prompt = dict(
        name = "CLIP",
        # pretrained = "./data/ViT-B-16.pt",
        embed_dim = 512,
        image_resolution = 224,
        vision_layers = 12,
        vision_width = 768,
        vision_patch_size = 16,
        context_length = 77,
        vocab_size = 49408,
        transformer_width = 512,
        transformer_heads = 8,
        transformer_layers = 12,
        joint=False,
        tsm=False,
        clip_seg_num=clip_seg_num,
        dropout = 0.,
        emb_dropout = 0.,
    ),
    text_prompt = dict(
        name = "TextCLIP"
    ),
    fusion_neck = dict(
        name = "ActionCLIPFusionNeck",
        sim_head = "Transf",
        embed_dim_cfg = 512,
        context_length_cfg = 77,
        transformer_width_cfg = 512,
        clip_seg_num = clip_seg_num
    ),
    head = dict(
        name = "MultiStageModel",
        num_stages = 4,
        num_layers = 10,
        num_f_maps = 64,
        dim = 512,
        num_classes = 11,
        sample_rate = sample_rate
    ),
    aligin_head = dict(
        name = "InterploteAlignHead"
    ),
    loss = dict(
        name = "ActionCLIPSegmentationLoss",
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