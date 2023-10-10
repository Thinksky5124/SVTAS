'''
Author       : Thyssen Wen
Date         : 2022-11-09 16:16:34
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 11:03:23
Description  : file content
FilePath     : /SVTAS/config/_base_/dataloader/dataset/gtea/gtea_stream_compressed_video.py
'''
sample_rate = 1
gop_size=16
flow_clip_seg_num = 128
flow_sliding_window = 128
rgb_clip_seg_num = flow_clip_seg_num // gop_size
rgb_sliding_window = flow_sliding_window

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 2,
    num_workers = 2 * 2,
    train = dict(
        name = "CompressedVideoStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        sliding_window = 64,
        need_residual = True,
        need_mvs = True
    ),
    test = dict(
        name = "CompressedVideoStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = 64,
        need_residual = True,
        need_mvs = True
    )
)

METRIC = dict(
    TAS = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/gtea/mapping.txt",
    file_output = False,
    score_output = False),
)

DATASETPIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            name = "ThreePathwayVideoDecoder",
            rgb_backend=dict(
                    name='DecordContainer'),
            flow_backend=dict(
                name='DecordContainer',
                to_ndarray=True,
                sample_dim=2),
            res_backend=dict(
                    name='DecordContainer'),
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = True,
            sample_rate_dict={"imgs":sample_rate * gop_size, "flows":sample_rate, "res":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"imgs":rgb_clip_seg_num, "flows":flow_clip_seg_num, "res":flow_clip_seg_num, "labels":flow_clip_seg_num},
            sliding_window_dict={"imgs":rgb_sliding_window, "flows":flow_sliding_window, "res":flow_sliding_window, "labels":flow_sliding_window},
            sample_add_key_pair={"rgb_frames":"imgs", "flow_frames":"flows", "res_frames":"res"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "CompressedVideoStreamTransform",
            rgb = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ],
            flow = [
                dict(XToTensor = None),
                dict(ToFloat = None),
                dict(TensorPermute = dict(permute_list = [2, 0, 1])),
                dict(TensorImageResize = dict(size = 256)),
                dict(TensorCenterCrop = dict(crop_size = 224)),
                dict(ScaleTo1_1 = None)
            ],
            res = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(NormalizeColorTo1 = None)
            ]
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            name = "ThreePathwayVideoDecoder",
            rgb_backend=dict(
                    name='DecordContainer'),
            flow_backend=dict(
                name='DecordContainer',
                to_ndarray=True,
                sample_dim=2),
            res_backend=dict(
                    name='DecordContainer'),
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate_dict={"imgs":sample_rate * gop_size, "flows":sample_rate, "res":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"imgs":rgb_clip_seg_num, "flows":flow_clip_seg_num, "res":flow_clip_seg_num, "labels":flow_clip_seg_num},
            sliding_window_dict={"imgs":rgb_sliding_window, "flows":flow_sliding_window, "res":flow_sliding_window, "labels":flow_sliding_window},
            sample_add_key_pair={"rgb_frames":"imgs", "flow_frames":"flows", "res_frames":"res"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "CompressedVideoStreamTransform",
            rgb = [
                dict(ResizeImproved = dict(size = 256)),
                dict(CenterCrop = dict(size = 224)),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ],
            flow = [
                dict(XToTensor = None),
                dict(ToFloat = None),
                dict(TensorPermute = dict(permute_list = [2, 0, 1])),
                dict(TensorImageResize = dict(size = 256)),
                dict(TensorCenterCrop = dict(crop_size = 224)),
                dict(ScaleTo1_1 = None)
            ],
            res = [
                dict(ResizeImproved = dict(size = 256)),
                dict(CenterCrop = dict(size = 224)),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(NormalizeColorTo1 = None)
            ]
        )
    )
)