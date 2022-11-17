'''
Author       : Thyssen Wen
Date         : 2022-10-25 16:53:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-15 14:17:21
Description  : I3D Extractor Config
FilePath     : /SVTAS/config/extract/extract_feature/i3d_r50_flow_gtea.py
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
    backbone = dict(
        pretrained = "./data/checkpoint/slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth",
        depth=50,
        pretrained2d=False,
        in_channels=2,
        conv1_kernel=(1,7,7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False
    ),
    head = dict(
        input_seg_num = 10,
        in_channels = 2048,
        sample_rate = sample_rate,
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
    video_batch_size = 4,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/flow",
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
        backend = dict(
                name='DecordContainer',
                to_ndarray=True,
                sample_dim=2)
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
        clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
        sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
        sample_add_key_pair={"frames":"imgs"},
        channel_num_dict={"imgs":2},
        channel_mode_dict={"imgs":"XY"},
        sample_mode = "uniform",
    ),
    transform = dict(
        name = "VideoStreamTransform",
        transform_list = [
            dict(XToTensor = None),
            dict(ToFloat = None),
            dict(TensorPermute = dict(permute_list = [2, 0, 1])),
            dict(TensorImageResize = dict(size = 256)),
            dict(TensorCenterCrop = dict(crop_size = 224)),
            dict(ScaleTo1_1 = None)
        ]
    )
)