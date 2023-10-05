'''
Author       : Thyssen Wen
Date         : 2022-11-15 10:03:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-15 10:08:12
Description  : file content
FilePath     : /SVTAS/config/extract/extract_feature/i3d_r50_rgb_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py', '../../_base_/models/action_recognition/i3d_r50.py',
    '../../_base_/dataset/gtea/gtea_stream_video.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 1
clip_seg_num = 21

MODEL = dict(
    head = dict(
        input_seg_num = 3,
        in_channels = 2048,
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
    video_batch_size = 4,
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
        transform_list = [
            dict(ResizeImproved = dict(size = 256)),
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(TensorCenterCrop = dict(crop_size = 224)),
            dict(ScaleTo1_1 = None)
        ]
    )
)