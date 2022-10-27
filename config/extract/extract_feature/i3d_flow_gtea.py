'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:53:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 15:42:20
Description  : I3D Extractor Config
FilePath     : /SVTAS/config/extract_feature/i3d_flow_gtea.py
'''

_base_ = [
    '../../_base_/collater/stream_compose.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 1
clip_seg_num = 64

MODEL = dict(
    architecture = "Recognition3D",
    backbone = dict(
        name = "I3D",
        pretrained = "./data/i3d_flow.pt",
        in_channels = 2
    ),
    neck = None,
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 1024,
        input_seg_num = 8,
        output_seg_num = 1,
        sample_rate = sample_rate,
        pool_space = True,
        in_format = "N,C,T,H,W",
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
    video_batch_size = 4,
    num_workers = 2,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/flow",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window,
        clip_seg_num = clip_seg_num,
        sample_rate = sample_rate
    )
)

PIPELINE = dict(
    name = "BasePipline",
    decode = dict(
        name = "FlowVideoDecoder",
        backend = "numpy"
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        sliding_window = sliding_window,
        sample_mode = "uniform",
        channel_mode = "XY"
    ),
    transform = dict(
        name = "VideoStreamTransform",
        transform_list = [
            dict(FeatureToTensor = None),
            dict(ToFloat = None),
            dict(TensorPermute = dict(permute_list = [2, 0, 1])),
            dict(TensorImageResize = dict(size = 256)),
            dict(TensorCenterCrop = dict(crop_size = 224)),
            dict(ScaleTo1_1 = None)
        ]
    )
)