'''
Author       : Thyssen Wen
Date         : 2023-10-18 20:27:05
LastEditors  : Thyssen Wen
LastEditTime : 2023-12-12 15:48:48
Description  : file content
FilePath     : /SVTAS/config/profiling/svtas_rl_profiling.py
'''
_base_ = [
    '../_base_/dataloader/collater/batch_compose.py',
    '../_base_/logger/python_logger.py',
]

num_classes = 11
sample_rate = 2
clip_seg_num = 64
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 80

model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

ENGINE = dict(
    name = "TorchStandaloneProfilerEngine",
    record = dict(
        name = "ValueRecord"
    ),
    iter_method = dict(
        name = "IterMethod",
        iter_num = 50,
        batch_size = 1,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    ),
    torch_profile_cfg = {},
    # extra_profiler = dict(
    #     ds = dict(
    #         name = "DeepspeedProfiler",
    #     )
    # )
)

MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    # grad_accumulate = dict(
    #     name = "GradAccumulate",
    #     accumulate_type = "conf"
    # ),
    model = dict(
        name = "StreamVideoSegmentation",
        architecture_type ='3d',
        addition_loss_pos = 'with_backbone_loss',
        backbone = dict(
            name = "SwinTransformer3D",
            pretrained = "./data/checkpoint/swin_base_patch244_window877_kinetics600_22k.pth",
            pretrained2d = False,
            patch_size = [2, 4, 4],
            embed_dim = 128,
            depths = [2, 2, 18, 2],
            num_heads = [4, 8, 16, 32],
            window_size = [8,7,7],
            mlp_ratio = 4.,
            qkv_bias = True,
            qk_scale = None,
            drop_rate = 0.,
            attn_drop_rate = 0.,
            drop_path_rate = 0.2,
            patch_norm = True,
            # graddrop_config={"gd_downsample": 1, "with_gd": [[1, 1], [1, 1], [1] * 14 + [0] * 4, [0, 0]]}
        ),
        neck = dict(
            name = "TaskFusionPoolNeck",
            num_classes=num_classes,
            in_channels = 1024,
            clip_seg_num = clip_seg_num // 2,
            need_pool = True
        ),
        head = dict(
            name = "BRTSegmentationHead",
            num_head=1,
            state_len=512,
            causal=False,
            num_decoders=3,
            encoder_num_layers=8,
            decoder_num_layers=8,
            num_f_maps=128,
            dropout=0.5,
            input_dim=1024,
            num_classes=num_classes,
            channel_masking_rate=0.2,
            sample_rate=sample_rate * 2
        )
    ),
    post_processing = dict(
        name = "StreamScorePostProcessing",
        sliding_window = sliding_window,
        ignore_index = ignore_index
    ),
    criterion = dict(
        name = "StreamSegmentationLoss",
        backbone_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate * 2,
            smooth_weight = 0.0,
            ignore_index = -100
        ),
        head_loss_cfg = dict(
            name = "RLPGSegmentationLoss",
            num_classes = num_classes,
            smooth_weight = 0.15,
            sample_rate = sample_rate,
            ignore_index = ignore_index
        )
    ),
    optimizer = dict(
        name = "AdamWOptimizer",
        learning_rate = 0.0005,
        weight_decay = 1e-4,
        betas = (0.9, 0.999),
        finetuning_scale_factor=0.02,
        no_decay_key = [],
        finetuning_key = ["backbone."],
        freeze_key = [],
    ),
    lr_scheduler = dict(
        name = "CosineAnnealingLR",
        T_max = epochs,
        eta_min = 0.00001,
    )
)

DATALOADER = dict(
    name = "RandomTensorTorchDataloader",
    iter_num = 10,
    tensor_dict = dict(
        imgs = dict(
            shape = [1, clip_seg_num, 3, 224, 244],
            dtype = "float32"),
        masks = dict(
            shape = [1, clip_seg_num * sample_rate],
            dtype = "float32"),
        labels =dict(
            shape = [1, clip_seg_num * sample_rate],
            dtype = "int64"),
    )
)