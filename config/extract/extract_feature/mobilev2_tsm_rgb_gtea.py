'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:53:39
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-13 14:35:15
Description  : TSM Config
FilePath     : /SVTAS/config/extract/extract_feature/mobilev2_tsm_rgb_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 256
clip_seg_num = 256
output_dir_name = 'extract_features'

MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "MobileNetV2TSM",
        # pretrained = "./data/checkpoint/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
        # pretrained2d = True,
        # pretrained = "./data/checkpoint/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth",
        clip_seg_num = clip_seg_num,
        shift_div = 8,
        out_indices = (7, )
    ),
    neck = None,
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 1280,
        input_seg_num = clip_seg_num,
        output_seg_num = sliding_window,
        sample_rate = sample_rate,
        pool_space = True,
        in_format = "N*T,C,H,W",
        out_format = "NCT"
    ),
    loss = None
)

PRETRAINED = "./output/TSM_FC_256x2_gtea_split1/TSM_FC_256x2_gtea_split1_best.pt"

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 4,
    num_workers = 2,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window,
    )
)

PIPELINE = dict(
    name = "BaseDatasetPipline",
    decode = dict(
        name = "VideoDecoder",
        backend = dict(name="DecordContainer")
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
                mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
            ))
        ])
    )
)