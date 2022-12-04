'''
Author       : Thyssen Wen
Date         : 2022-11-19 11:15:20
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-22 22:48:17
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/tsm_fc_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_video.py'
]

num_classes = 11
sample_rate = 2
clip_seg_num = 32
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 2
epochs = 50

model_name = "TSM_FC_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "MobileNetV2TSM",
        pretrained = "./data/checkpoint/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
        clip_seg_num = clip_seg_num,
        shift_div = 8,
        out_indices = (7, ),
    ),
    # backbone = dict(
    #     name = "ResNetTSM",
    #     pretrained = "./data/checkpoint/resnet50-0676ba61.pth",
    #     depth=50,
    #     clip_seg_num = 8,
    #     shift_div = 8,
    #     norm_eval=False,
    #     torchvision_pretrain=True,
    # ),
    neck = dict(
        name = "PoolNeck",
        num_classes = num_classes,
        in_channels = 1280,
        clip_seg_num = clip_seg_num,
        need_pool = True
    ),
    head = dict(
        name = "FCHead",
        num_classes = num_classes,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        drop_ratio=0.5,
        in_channels=1280
        # name = "MultiStageModel",
        # num_stages = 1,
        # num_layers = 4,
        # num_f_maps = 64,
        # dim = 1280,
        # num_classes = num_classes,
        # sample_rate = sample_rate
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 1.0,
        ignore_index = -100
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

OPTIMIZER = dict(
    learning_rate = 0.0005,
    weight_decay = 1e-5,
    betas = (0.9, 0.999),
    need_grad_accumulate = True,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = [],
    freeze_key = [],
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = batch_size,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    )
)

METRIC = dict(
    TAS = dict(
    file_output = False,
    score_output = False),
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
