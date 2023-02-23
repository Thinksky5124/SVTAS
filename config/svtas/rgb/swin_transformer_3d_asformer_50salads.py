'''
Author       : Thyssen Wen
Date         : 2022-12-18 19:04:09
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-23 12:55:18
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/swin_transformer_3d_asformer_50salads.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/action_recognition/swin_transformer.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/50salads/50salads_stream_video.py'
]

num_classes = 19
sample_rate = 4
clip_seg_num = 64
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 50
log_interval = 10
save_interval = 1

model_name = "SwinTransformer3D_Asformer_"+str(clip_seg_num)+"x"+str(sample_rate)+"_50salads_split" + str(split)

MODEL = dict(
    architecture = "StreamSegmentation3DWithBackbone",
    backbone = dict(
        name = "SwinTransformer3D",
        pretrained = "./data/checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth",
        pretrained2d = False,
        patch_size = [2, 4, 4],
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = [8,7,7],
        mlp_ratio = 4.,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.2,
        patch_norm = True,
        # sbp_build=True,
        # keep_ratio_list=[0.125],
        # sample_dims=[2],
        # grad_mask_mode_lsit=['random'],
        # register_sbp_module_dict={Mlp: Swin3DMLPMaskMappingFunctor(permute_dims=[0, 2, 3, 4, 1])}
    ),
    neck = dict(
        name = "TaskFusionPoolNeck",
        num_classes=num_classes,
        in_channels = 768,
        clip_seg_num = clip_seg_num // 2,
        need_pool = True,
        fusion_ratio = 0.0
    ),
    head = dict(
        name = "ASFormer",
        num_decoders = 3,
        num_layers = 10,
        r1 = 2,
        r2 = 2,
        num_f_maps = 64,
        input_dim = 768,
        channel_masking_rate = 0.5,
        num_classes = num_classes,
        sample_rate = sample_rate * 2
    ),
    loss = dict(
        name = "StreamSegmentationLoss",
        backbone_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate * 2,
            smooth_weight = 0.0,
            ignore_index = -100
        ),
        head_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate,
            smooth_weight = 0.0,
            ignore_index = -100
        )

        # name = "DiceSegmentationLoss",
        # # class_weight = [0.40501603,1.7388232,0.5236841,2.3680801,0.52725035,1.8183347,
        # #                 2.1976302,1.0866599,0.9076069,1.8409629,1.1957755,0.403674,0.5133538,
        # #                 1.5752678,1.1706547,1.0,0.7277812,0.8284057,0.48404875],
        # num_classes = num_classes,
        # sample_rate = sample_rate * 2,
        # smooth_weight = 0.0,
        # ignore_index = -100
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
    learning_rate = 0.0001,
    weight_decay = 1e-4,
    betas = (0.9, 0.999),
    need_grad_accumulate = False,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = ["backbone"],
    freeze_key = [],
)

METRIC = dict(
    TAS = dict(
        file_output = False,
        score_output = False),
    ACC = dict(
        name = "ConfusionMatrix",
        actions_map_file_path = "./data/50salads/mapping.txt",
        img_save_path = "./output",
        need_plot = False,
        need_color_bar = False,),
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/50salads/Videos_mp4",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/50salads/Videos_mp4",
        sliding_window = sliding_window,
    )
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
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                ))]
            )
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
            transform_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 224)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                        std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                    ))]
            )
        )
    )
)
