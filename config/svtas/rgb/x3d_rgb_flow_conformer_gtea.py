'''
Author       : Thyssen Wen
Date         : 2022-11-14 21:16:31
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-14 21:46:03
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/x3d_rgb_flow_conformer_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adam.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_rgb_flow_video.py'
]
split = 1
num_classes = 11
sample_rate = 1
gop_size=16
flow_clip_seg_num = 128
flow_sliding_window = 128
rgb_clip_seg_num = flow_clip_seg_num // gop_size
rgb_sliding_window = flow_sliding_window

ignore_index = -100
batch_size = 2
epochs = 50
model_name = "X3D_Flow_Rgb_Freeze_IPB_Conformer_128x1_gtea_split" + str(split)

MODEL = dict(
    architecture = "MultiModalityStreamSegmentation",
    rgb_backbone_type='2d',
    flow_backbone_type='2d',
    rgb_backbone = dict(
        name = "MobileNetV2TSM",
        pretrained = "./data/checkpoint/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth",
        clip_seg_num = rgb_clip_seg_num,
        shift_div = 8,
        out_indices = (7, )
    ),
    flow_backbone = dict(
        name = "MobileNetV2TSM",
        # pretrained = "./data/checkpoint/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth",
        clip_seg_num = flow_clip_seg_num,
        in_channels = 2,
        shift_div = 8,
        out_indices = (7, )
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
        clip_seg_num = flow_clip_seg_num,
        fusion_mode ='stack',
        fusion_neck_module = dict(
            name = "AvgPoolNeck",
            num_classes = num_classes,
            in_channels = 384,
            clip_seg_num = flow_clip_seg_num,
            drop_ratio = 0.5,
            need_pool = True
        )
    ),
    head = dict(
        name = "Conformer",
        num_classes = num_classes,
        sample_rate = sample_rate,
        input_dim = 384,
        encoder_dim = 64,
        num_stages = 3,
        num_encoder_layers = 1,
        input_dropout_p = 0.5,
        num_attention_heads = 8,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 7,
        half_step_residual = True,
        need_subsampling = False,
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
    freeze_key = [],
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