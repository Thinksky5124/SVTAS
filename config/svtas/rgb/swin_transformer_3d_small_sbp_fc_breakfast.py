'''
Author       : Thyssen Wen
Date         : 2022-12-18 19:04:09
LastEditors  : Thyssen Wen
LastEditTime : 2023-03-30 09:54:51
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/swin_transformer_3d_small_sbp_fc_breakfast.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/cosine_50e.py',
    '../../_base_/models/action_recognition/swin_transformer.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/breakfast/breakfast_stream_video.py'
]
num_classes = 48
sample_rate = 4
clip_seg_num = 128
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 50
log_interval = 100

model_name = "SwinTransformer3DSBP_FC_"+str(clip_seg_num)+"x"+str(sample_rate)+"_breakfast_split" + str(split)

MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "SwinTransformer3DWithSBP",
        pretrained = "./data/checkpoint/swin_small_patch244_window877_kinetics400_1k.pth",
        pretrained2d = False,
        patch_size = [2, 4, 4],
        embed_dim = 96,
        depths = [2, 2, 18, 2],
        num_heads = [3, 6, 12, 24],
        window_size = [8,7,7],
        mlp_ratio = 4.,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.2,
        patch_norm = True,
        graddrop_config={"gd_downsample": 1, "with_gd": [[1, 1], [1, 1], [1] * 14 + [0] * 4, [0, 0]]}
    ),
    neck = dict(
        name = "PoolNeck",
        in_channels = 768,
        clip_seg_num = clip_seg_num // 2,
        need_pool = True
    ),
    head = dict(
        name = "FCHead",
        num_classes = num_classes,
        sample_rate = sample_rate * 2,
        clip_seg_num = clip_seg_num // 2,
        drop_ratio=0.5,
        in_channels=768
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate * 2,
        smooth_weight = 0.0,
        ignore_index = -100
    ) 
)

POSTPRECESSING = dict(
    name = "StreamScorePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

LRSCHEDULER = dict(
    name = "CosineAnnealingLR",
    T_max = epochs,
    eta_min = 0.000001,
)

OPTIMIZER = dict(
    learning_rate = 0.00001,
    weight_decay = 1e-4,
    betas = (0.9, 0.999),
    need_grad_accumulate = False,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = [],
    freeze_key = [],
)

# METRIC = dict(
#     TAS = dict(
#         file_output = True,
#         score_output = True),
#     ACC = dict(
#         name = "ConfusionMatrix",
#         actions_map_file_path = "./data/breakfast/mapping.txt",
#         img_save_path = "./output",
#         need_plot = True,
#         need_color_bar = False,)
# )

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/breakfast/Videos_mp4",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle",
        videos_path = "./data/breakfast/Videos_mp4",
        sliding_window = sliding_window,
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
            name = "VideoStreamSampler",
            is_train = True,
            sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"imgs"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "VideoTransform",
            transform_dict = dict(
                imgs = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))]
            )
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
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"imgs"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "VideoTransform",
            transform_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 224)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                    ))]
            )
        )
    )
)
