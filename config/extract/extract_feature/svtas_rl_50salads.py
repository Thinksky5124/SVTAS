'''
Author       : Thyssen Wen
Date         : 2023-11-03 10:37:38
LastEditors  : Thyssen Wen
LastEditTime : 2023-11-03 10:51:46
Description  : file content
FilePath     : /SVTAS/config/extract/extract_feature/svtas_rl_50salads.py
'''
_base_ = [
    '../../_base_/dataloader/collater/batch_compose.py',
    '../../_base_/logger/python_logger.py',
]

sample_rate = 8
batch_size = 1
ignore_index = -100
clip_seg_num = 128
sliding_window = clip_seg_num * sample_rate
num_classes = 19

model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_50salads"

ENGINE = dict(
    name = "ExtractFeatureEngine",
    out_path = "data/50salads/extract_feature",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = 1,
        batch_size = batch_size,
        logger_iter_interval = 10,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    )
)

MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    pretrained = "/share/final2_RGB_50saladsmcepoch80SwinTransformer3D_BRT_128x8_50salads_split1_best.pt",
    model = dict(
        name = "StreamVideoSegmentation",
        architecture_type ='3d',
        addition_loss_pos = 'with_backbone_loss',
        backbone = dict(
            name = "SwinTransformer3DWithSBP",
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
            graddrop_config={"gd_downsample": 1, "with_gd": [[1, 1], [1, 1], [1] * 14 + [0] * 4, [0, 0]]}
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
        name = "StreamFeaturePostProcessing",
        sliding_window = sliding_window,
        ignore_index = ignore_index
    ),
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/50salads/tiny.txt",
        videos_path = "./data/50salads/Videos",
        gt_path = "./data/50salads/groundTruth",
        actions_map_file_path = "./data/50salads/mapping.txt",
        dataset_type = "50salads",
        train_mode = False,
        sliding_window = sliding_window
    )
)

DATASETPIPLINE = dict(
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
                    mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                    std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
                ))],
            masks = dict(
                masks = dict(name = 'direct_transform',
                                transforms_op_list = [
                                    dict(NumpyDataTypeTransform = dict(dtype = "float32"))])
            )
        )
    )
)