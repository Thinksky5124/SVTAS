'''
Author       : Thyssen Wen
Date         : 2023-02-18 19:31:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-18 19:48:21
Description  : file content
FilePath     : /SVTAS/config/extract/extract_feature/resnet_rgb_50salads.py
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
        name = "ResNet",
        pretrained = "./data/checkpoint/resnet50-0676ba61.pth",
        depth = 50,
    ),
    neck = None,
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 1024,
        input_seg_num = clip_seg_num,
        output_seg_num = clip_seg_num,
        sample_rate = sample_rate,
        pool_space = True,
        in_format = "N*T,C,H,W",
        out_format = "NCT"
    ),
    loss = None
)

PRETRAINED = None

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 1,
    num_workers = 2,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/50salads/splits/all_files.txt",
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
                mean = [0.5139909998345553 * 255, 0.5117725498677757 * 255, 0.4798814301515671 * 255],
                std = [0.23608918491478523 * 255, 0.23385714300069754 * 255, 0.23755006337414028* 255]
            ))]
        )
    )
)