'''
Author       : Thyssen Wen
Date         : 2022-11-19 11:14:54
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-19 14:44:45
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/i3d_r50_fc_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
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

model_name = "I3D_R50_FC_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "ResNet3d",
        pretrained = "./data/checkpoint/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth",
        in_channels=3,
        pretrained2d=False,
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=1,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False,
        with_pool1=True,
        with_pool2=True,
    ),
    neck = dict(
        name = "PoolNeck",
        num_classes = num_classes,
        in_channels = 2048,
        clip_seg_num = clip_seg_num // 8,
        need_pool = True
    ),
    head = dict(
        # name = "FCHead",
        # num_classes = num_classes,
        # sample_rate = sample_rate * 8,
        # clip_seg_num = clip_seg_num // 8,
        # drop_ratio=0.5,
        # in_channels=2048
        name = "MultiStageModel",
        num_stages = 1,
        num_layers = 4,
        num_f_maps = 64,
        dim = 2048,
        num_classes = num_classes,
        sample_rate = sample_rate * 8
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
    learning_rate = 0.001,
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
    file_output = True,
    score_output = True
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
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"imgs"},
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
