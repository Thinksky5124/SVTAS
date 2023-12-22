'''
Author       : Thyssen Wen
Date         : 2023-10-18 20:27:05
LastEditors  : Thyssen Wen
LastEditTime : 2023-12-20 10:39:10
Description  : file content
FilePath     : /SVTAS/config/profiling/svtas_rl_profiling_numerical_range.py
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

from svtas.model.video.swin_transformer_3d import Mlp
from svtas.model.tas.block_recurrent_transformer.brt_segmenter import RecurrentAttentionBlock

model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_numerical_range_gtea_split" + str(split)

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
    torch_profile_cfg = None,
    extra_profiler = dict(
        nr = dict(
            name = "NumericalRangeProfiler",
            profile_step = 5,
            profile_layers = [RecurrentAttentionBlock],
            profile_activation = True,
            profile_weight = True,
            need_plot = True,
            plot_path = "./output"
        )
    )
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
    name = "TorchStreamDataLoader",
    batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    test = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window,
    )
)

DATASETPIPLINE = dict(
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
            transform_results_list = [
                dict(DropResultsByKeyName = dict(drop_keys_list=[
                    "filename", "raw_labels", "sample_sliding_idx", "format", "frames", "frames_len", "feature_len", "video_len"
                ])),
                dict(RenameResultTransform = dict(rename_pair_dict=dict(
                    video_name = "vid_list"
                )))
            ],
            transform_key_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 224)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                    ))],
                masks = dict(
                    masks = dict(name = 'direct_transform',
                                 transforms_op_list = [
                                     dict(NumpyDataTypeTransform = dict(dtype = "float32"))])
                )
            )
        )
    )
)