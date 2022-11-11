'''
Author       : Thyssen Wen
Date         : 2022-11-04 09:35:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-04 10:06:01
Description  : file content
FilePath     : /SVTAS/config/extract/extract_feature/x3d_rgb_gtea.py
'''

_base_ = [
    '../../_base_/collater/stream_compose.py', '../../_base_/models/action_recognition/x3d.py',
    '../../_base_/dataset/gtea/gtea_stream_video.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 1
clip_seg_num = 21
batch_size = 4

MODEL = dict(
    backbone = dict(
        name = "X3D",
        pretrained="data/checkpoint/x3d_m.pyth",
        dim_c1=12,
        scale_res2=False,
        depth=50,
        num_groups=1,
        width_per_group=64,
        width_factor=2.0,
        depth_factor=2.2,
        input_channel_num=[3],
        bottleneck_factor=2.25,
        channelwise_3x3x3=True
    ),
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 192,
        input_seg_num = clip_seg_num,
        output_seg_num = 1,
        sample_rate = sample_rate,
        pool_space = True,
        in_format = "N,C,T,H,W",
        out_format = "NCT"
    )
)

PRETRAINED = None

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    video_batch_size = batch_size,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    )
)

PIPELINE = dict(
    name = "BasePipline",
    decode = dict(
        name = "VideoDecoder",
        backend = "decord"
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
        name = "VideoStreamTransform",
        transform_list = [
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(RandomShortSideScaleJitter = dict(min_size=256, max_size=320)),
            dict(TensorCenterCrop = dict(crop_size = 224)),
            dict(NormalizeColorTo1 = None),
            dict(Normalize = dict(
                mean = [140.39158961711036 / 255.0, 108.18022223151027 / 255.0, 45.72351736766547 / 255.0],
                std = [33.94421369129452 / 255.0, 35.93603536756186 / 255.0, 31.508484434367805 / 255.0]
            ))
        ]
    )
)