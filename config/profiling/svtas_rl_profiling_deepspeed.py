'''
Author       : Thyssen Wen
Date         : 2023-10-18 20:27:05
LastEditors  : Thyssen Wen
LastEditTime : 2023-12-14 10:57:45
Description  : file content
FilePath     : /SVTAS/config/profiling/svtas_rl_profiling_deepspeed.py
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

# model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_profile_gtea_split" + str(split)
model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_profile_compression_gtea_split" + str(split)

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
    torch_profile_cfg = dict(
        wait = 10,
        warmup = 5,
    ),
    # extra_profiler = dict(
    #     ds = dict(
    #         name = "DeepspeedProfiler",
    #     )
    # )
)

MODEL_PIPLINE = dict(
    name = "DeepspeedModelPipline",
    # grad_accumulate = dict(
    #     name = "GradAccumulate",
    #     accumulate_type = "conf"
    # ),
    compression = True,
    ds_config = dict(
        train_micro_batch_size_per_gpu = batch_size,
        wall_clock_breakdown = False,
        compression_training = {
            "layer_reduction": {
                "enabled": False,
                "keep_number_layer": 5,
                "module_name_prefix": "bert.encoder.layer",
                "teacher_layer": [
                    2,
                    4,
                    6,
                    8,
                    10
                ],
                "other_module_name": [
                    "bert.pooler",
                    "bert.embeddings",
                    "classifier"
                ]
            },
            "weight_quantization": {
                "shared_parameters": {
                    "enabled": True,
                    "quantizer_kernel": False,
                    "schedule_offset": 0,
                    "quantize_groups": 48,
                    "quantize_verbose": True,
                    "quantization_type": "symmetric",
                    "quantize_weight_in_forward": True,
                    "rounding": "nearest",
                    "fp16_mixed_quantize": {
                        "enabled": False,
                        "quantize_change_ratio": 0.1
                    }
                },
                "different_groups": {
                    "wq1": {
                        "params": {
                            "start_bits": 8,
                            "target_bits": 8,
                            "quantization_period": 0
                        },
                        "modules": [
                            "All Linear and CONV2D layers"
                        ]
                    }
                }
            },
            "activation_quantization": {
                "shared_parameters": {
                "enabled": True,
                "quantization_type": "symmetric",
                "range_calibration": "dynamic",
                "schedule_offset": 0
                },
                "different_groups": {
                    "aq1": {
                            "params": {
                            "bits": 8
                        },
                        "modules": [
                            "All Linear and CONV2D layers"
                        ]
                    }
                }
            },
            "sparse_pruning": {
                "shared_parameters": {
                "enabled": False,
                "schedule_offset": 2000,
                "method": "l1"
                },
                "different_groups": {
                    "sp1": {
                        "params": {
                            "dense_ratio": 0.5
                        },
                        "modules": [
                            "attention.self"
                        ]
                    }
                }
            },
            "row_pruning": {
                "shared_parameters": {
                    "enabled": False,
                    "schedule_offset": 2000,
                    "method": "topk"
                },
                "different_groups": {
                    "rp1": {
                        "params": {
                            "dense_ratio": 0.5
                        },
                        "modules": [
                            "intermediate.dense"
                        ],
                        "related_modules": [
                            [
                                "layer.\\w+.output.dense"
                            ]
                        ]
                    }
                }
            },
            "head_pruning": {
                "shared_parameters": {
                "enabled": False,
                "schedule_offset": 2000,
                "method": "topk",
                "num_heads": 12
                },
                "different_groups": {
                "rp1": {
                    "params": {
                        "dense_ratio": 0.5
                    },
                    "modules": [
                        "attention.output.dense"
                    ],
                    "related_modules": [
                        [
                            "self.query",
                            "self.key",
                            "self.value"
                        ]
                    ]
                }
                }
            }
        }
    ),
    pretrained = "output/SVTAS-RL_64x2_compression_gtea_split1/2023-12-14-10-24-59/ckpt/SVTAS-RL_64x2_compression_gtea_split1_best/mp_rank_00_model_states.pt",
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
    iter_num = 50,
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