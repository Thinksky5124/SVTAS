_base_ = [
    '../_base_/models/image_classification/swin_v2_transformer.py',
    '../_base_/default_runtime.py', '../_base_/collater/stream_compose.py',
]

num_classes = 11
sample_rate = 2
clip_seg_num = 32
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 2
epochs = 50

MODEL = dict(
    architecture = "Recognition2D",
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
        clip_seg_num = clip_seg_num,
        need_pool = True
    ),
    head = dict(
        name = "FCHead",
        num_classes = num_classes,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        drop_ratio=0.5,
        in_channels=768
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 0.0,
        ignore_index = ignore_index
    )        
)

PRETRAINED = "./output/SwinTransformerV2_FC_32x2_gtea_split1/SwinTransformerV2_FC_32x2_gtea_split1_best.pt"

VISUALIZE = dict(
    layer_name = ["model.backbone.norm"],
    batch_size = batch_size,
    sample_rate = sample_rate,
    ignore_index = ignore_index,
    data_key = "imgs",
    return_targets_name = dict(
        CategorySegmentationTarget = dict(category=None)
    ),
    reshape_transform = "NPC",
    label_path = "./data/gtea/mapping.txt",
    match_fn_name = "rgb_stream_match_fn"
)

POSTPRECESSING = dict(
    name = "CAMVideoPostProcessing",
    sample_rate=sample_rate,
    ignore_index=ignore_index,
    fps=4,
    output_frame_size = [720, 404],
    need_label=True
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2,
    config = dict(
        name = "RawFrameStreamCAMDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    ),
)

PIPELINE = dict(
    name = "BasePipline",
    decode = dict(
        name="VideoDecoder",
        backend=dict(
                name='DecordContainer')
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate_dict={"imgs":sample_rate, "labels":sample_rate},
        clip_seg_num_dict={"imgs":clip_seg_num, "labels":clip_seg_num},
        sliding_window_dict={"imgs":sliding_window, "labels":sliding_window},
        sample_add_key_pair={"frames":"imgs"},
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "VideoRawStoreTransform",
        transform_dict = dict(
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