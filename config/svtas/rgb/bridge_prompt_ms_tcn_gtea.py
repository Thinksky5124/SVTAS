'''
Author       : Thyssen Wen
Date         : 2022-10-30 16:48:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-05 21:16:48
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/bridge_prompt_ms_tcn_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_video.py'
]
split = 1
num_classes = 11
sample_rate = 1
clip_seg_num = 32
sliding_window = 32
ignore_index = -100
batch_size = 1
epochs = 50
cnt_max = 30
model_name = "BridgePrompt_gtea_split" + str(split)

MODEL = dict(
    architecture = "ActionCLIP",
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
    text_prompt = dict(
        name = "BridgePromptTextEncoder",
        actions_map_file_path = "./data/gtea/mapping.txt",
        cnt_max=cnt_max
    ),
    fusion_neck = dict(
        name = "BridgePromptFusionEarlyhyp",
        embedding_dim = 512,
        num_layers=6,
        cnt_max=cnt_max,
        clip_seg_num = clip_seg_num
    ),
    head = dict(
        name = "MultiStageModel",
        num_stages = 1,
        num_layers = 4,
        num_f_maps = 64,
        dim = 512,
        num_classes = 11,
        sample_rate = sample_rate
    ),
    loss = dict(
        name = "BridgePromptCLIPSegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        cnt_max = cnt_max,
        smooth_weight = 0.15,
        ignore_index = -100,
        is_segmentation=True
    )
)

POSTPRECESSING = dict(
    name = "StreamScorePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        sliding_window = sliding_window,
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
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            sliding_window = sliding_window,
            sample_mode = "uniform"
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
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            sliding_window = sliding_window,
            sample_mode = "uniform"
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
