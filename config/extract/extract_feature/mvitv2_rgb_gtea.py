'''
Author       : Thyssen Wen
Date         : 2022-10-30 15:37:30
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-30 16:32:41
Description  : MViTV2 Config
FilePath     : /SVTAS/config/extract/extract_feature/mvitv2_rgb_gtea.py
'''

_base_ = [
    '../../_base_/collater/stream_compose.py', '../../_base_/models/action_recognition/mvitv2_b.py',
    '../../_base_/dataset/gtea/gtea_stream_video.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 1
clip_seg_num = 32

MODEL = dict(
    backbone = dict(
        name = "MViT",
        clip_seg_num = 32,
        pretrained = "./data/MViTv2_B_32x3_k400_f304025456.pyth"
    ),
    head = dict(
        sample_rate = sample_rate
    )
)

PRETRAINED = None

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    video_batch_size = 1,
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
        sample_rate = 3,
        clip_seg_num = clip_seg_num,
        sliding_window = sliding_window,
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "VideoStreamTransform",
        transform_list = [
            dict(ResizeImproved = dict(size = 256)),
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(TensorCenterCrop = dict(crop_size = 224)),
            dict(ScaleTo1_1 = None)
        ]
    )
)