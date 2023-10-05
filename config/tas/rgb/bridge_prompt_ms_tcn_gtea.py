'''
Author       : Thyssen Wen
Date         : 2022-10-30 16:48:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 20:10:46
Description  : file content
FilePath     : /SVTAS/config/tas/rgb/bridge_prompt_ms_tcn_gtea.py
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
cnt_max = 30
model_name = "BridgePrompt_MS_TCN_gtea_split" + str(split)

MODEL = dict(
    architecture = "ActionCLIPSegmentation",
    pretrained = "./data/checkpoint/vit-16-32f.pt",
    image_prompt = dict(
        name = "CLIP",
        # pretrained = "./data/checkpoint/ViT-B-16.pt",
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
    text_prompt = None,
    fusion_neck = None,
    head = dict(
        name = "MultiStageModel",
        num_stages = 3,
        num_layers = 5,
        num_f_maps = 64,
        dim = 512,
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