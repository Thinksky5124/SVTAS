_base_ = [
    '../_base_/dataloader/collater/batch_compose.py',
    '../_base_/logger/python_logger.py',
]

num_classes = 11
clip_seg_num_list = [32, 40]
sample_rate_list = [2]
clip_seg_num = 32
sample_rate = 2
ignore_index = -100
split = 1
batch_size = 2

model_name = "SwinV2_FC_dynamic_CAM_gtea"

ENGINE = dict(
    name = "VisualEngine",
    out_path = "output/cam",
    label_path = "./data/gtea/mapping.txt",
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
    name = "TorchCAMModelPipline",
    layer_name = ["model.backbone.norm"],
    batch_size = batch_size,
    sample_rate = sample_rate,
    ignore_index = ignore_index,
    data_key = "imgs",
    return_targets_name = dict(
        CategorySegmentationTarget = dict(category=None)
    ),
    reshape_transform_name = "NPC",
    label_path = "./data/gtea/mapping.txt",
    match_fn = "rgb_stream_match_fn",
    method = "gradcam++",
    # pretrained = "",
    model = dict(
        name = "StreamVideoSegmentation",
        architecture_type ='2d',
        backbone = dict(
            name = "SwinTransformerV2",
            pretrained = "./data/checkpoint/swinv2_tiny_patch4_window8_256.pth",
            img_size=256,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            drop_path_rate=0.2,
        ), 
        neck = dict(
            name = "PoolNeck",
            in_channels = 768,
            need_pool = True
        ),
        head = dict(
            name = "FCHead",
            num_classes = num_classes,
            sample_rate = sample_rate,
            drop_ratio=0.5,
            in_channels=768
        ),
    ),
    post_processing = dict(
        name = "CAMVideoPostProcessing",
        sample_rate=sample_rate,
        ignore_index=ignore_index,
        fps=4,
        output_frame_size = [720, 404],
        need_label=True
    ),
    criterion = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 0.0,
        ignore_index = ignore_index
    ),
)

POSTPRECESSING = dict(
    name = "CAMVideoPostProcessing",
    sample_rate=sample_rate,
    ignore_index=ignore_index,
    fps=4,
    output_frame_size = [720, 404],
    need_label=True
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    config = dict(
        name = "RawFrameDynamicStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        dynamic_stream_generator=dict(
            name = "MultiEpochStageDynamicStreamGenerator",
            multi_epoch_list = [40],
            strategy_list = [
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [32, 40],
                     sample_rate_list = sample_rate_list),
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [16, 32],
                     sample_rate_list = sample_rate_list)
            ]
        )
    ),
)

DATASETPIPLINE = dict(
    name = "BaseDatasetPipline",
    decode = dict(
        name="VideoDecoder",
        backend=dict(
                name='DecordContainer')
    ),
    sample = dict(
        name = "VideoDynamicStreamSampler",
        is_train = True,
        sample_rate_name_dict={"imgs":'sample_rate', "labels":'sample_rate'},
        clip_seg_num_name_dict={"imgs": 'clip_seg_num', "labels": 'clip_seg_num'},
        ignore_index=ignore_index,
        sample_add_key_pair={"frames":"imgs"},
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "VideoRawStoreTransform",
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
            dict(CenterCrop = dict(size = 256)),
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(Normalize = dict(
                mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
            ))
        ])
    )
)