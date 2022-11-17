'''
Author       : Thyssen Wen
Date         : 2022-11-15 15:02:52
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-15 18:54:30
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/i3d_r50_rgb_flow_mstcn_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_rgb_flow_video.py'
]
split = 1
num_classes = 11
sample_rate = 4
gop_size = 1
flow_clip_seg_num = 512
flow_sliding_window = flow_clip_seg_num * sample_rate
rgb_clip_seg_num = flow_clip_seg_num // gop_size
rgb_sliding_window = flow_sliding_window

ignore_index = -100
batch_size = 2
epochs = 50
model_name = "I3D_R50_Flow_Rgb_Freeze_Asformer_512x1_gtea_split" + str(split)

MODEL = dict(
    architecture = "MultiModalityStreamSegmentation",
    rgb_backbone = dict(
        name = "ResNet3d",
        pretrained = "./data/checkpoint/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth",
        depth=50,
        pretrained2d=False,
        in_channels=3,
        conv1_kernel=(1,7,7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False
    ),
    flow_backbone = dict(
        name = "ResNet3d",
        pretrained = "./data/checkpoint/slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth",
        depth=50,
        pretrained2d=False,
        in_channels=2,
        conv1_kernel=(1,7,7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False
    ),
    # neck = dict(
    #     name = "IPBFusionNeck",
    #     gop_size = gop_size,
    #     spatial_expan_mode ='trilinear',
    #     parse_method ='separate',
    #     fusion_neck_module = dict(
    #         name = "AvgPoolNeck",
    #         num_classes = num_classes,
    #         in_channels = 2048,
    #         clip_seg_num = flow_clip_seg_num // 8,
    #         drop_ratio = 0.5,
    #         need_pool = True
    #     )
    # ),
    neck = dict(
        name = "MultiModalityFusionNeck",
        clip_seg_num = flow_clip_seg_num // 2,
        fusion_mode ='stack',
        fusion_neck_module = dict(
            name = "AvgPoolNeck",
            num_classes = num_classes,
            in_channels = 2048+2048,
            clip_seg_num = flow_clip_seg_num // 2,
            drop_ratio = 0.5,
            need_pool = True
        )
    ),
    head = dict(
        name = "ASFormer",
        num_decoders = 3,
        num_layers = 10,
        r1 = 2,
        r2 = 2,
        num_f_maps = 64,
        input_dim = 2048+2048,
        channel_masking_rate = 0.5,
        num_classes = num_classes,
        sample_rate = sample_rate * 2
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
    name = "StreamScorePostProcessing",
    sliding_window = flow_sliding_window,
    ignore_index = ignore_index
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

OPTIMIZER = dict(
    learning_rate = 0.0005,
    weight_decay = 1e-4,
    betas = (0.9, 0.999),
    need_grad_accumulate = True,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = [],
    freeze_key = ["rgb_backbone", "flow_backbone"],
)

DATASET = dict(
    video_batch_size = batch_size,
    num_workers = batch_size * 2,
    train = dict(
        sliding_window = flow_sliding_window,
    ),
    test = dict(
        sliding_window = flow_sliding_window,
    )
)

PIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            name = "TwoPathwayVideoDecoder",
            rgb_backend=dict(
                name='DecordContainer'),
            flow_backend=dict(
                name='DecordContainer',
                to_ndarray=True,
                sample_dim=2)
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = True,
            sample_rate_dict={"imgs":sample_rate * gop_size, "flows":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"imgs":rgb_clip_seg_num, "flows":flow_clip_seg_num, "labels":flow_clip_seg_num},
            sliding_window_dict={"imgs":rgb_sliding_window, "flows":flow_sliding_window, "labels":flow_sliding_window},
            sample_add_key_pair={"rgb_frames":"imgs", "flow_frames":"flows"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "RGBFlowVideoStreamTransform",
            rgb = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ],
            flow = [
                dict(XToTensor = None),
                dict(ToFloat = None),
                dict(TensorPermute = dict(permute_list = [2, 0, 1])),
                dict(TensorImageResize = dict(size = 256)),
                dict(TensorCenterCrop = dict(crop_size = 224)),
                dict(ScaleTo1_1 = None)
            ]
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            name = "TwoPathwayVideoDecoder",
            rgb_backend=dict(
                name='DecordContainer'),
            flow_backend=dict(
                name='DecordContainer',
                to_ndarray=True,
                sample_dim=2)
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate_dict={"imgs":sample_rate * gop_size, "flows":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"imgs":rgb_clip_seg_num, "flows":flow_clip_seg_num, "labels":flow_clip_seg_num},
            sliding_window_dict={"imgs":rgb_sliding_window, "flows":flow_sliding_window, "labels":flow_sliding_window},
            sample_add_key_pair={"rgb_frames":"imgs", "flow_frames":"flows"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "RGBFlowVideoStreamTransform",
            rgb = [
                dict(ResizeImproved = dict(size = 256)),
                dict(CenterCrop = dict(size = 224)),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ],
            flow = [
                dict(XToTensor = None),
                dict(ToFloat = None),
                dict(TensorPermute = dict(permute_list = [2, 0, 1])),
                dict(TensorImageResize = dict(size = 256)),
                dict(TensorCenterCrop = dict(crop_size = 224)),
                dict(ScaleTo1_1 = None)
            ]
        )
    )
)
