'''
Author       : Thyssen Wen
Date         : 2022-11-11 09:31:23
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-11 20:35:04
Description  : file content
FilePath     : /SVTAS/config/extract/extract_mvs_res/mvs_res_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_compressed_video.py'
]

sliding_window = 64
sample_rate = 1

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 2,
    num_workers = 4,
    config = dict(
        name = "CompressedVideoStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window,
        need_residual = True,
        need_mvs = True
    )
)

POSTPRECESSING = dict(
    name = "MVsResPostProcessing",
    fps = 15,
    need_visualize = False,
    sliding_window = sliding_window
)

PIPELINE = dict(
    name = "BasePipline",
    decode = dict(
        name = "VideoDecoder",
        backend=dict(
                name='PyAVMVExtractor',
                need_residual=True,
                need_mvs=True,
                argument=False)
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate_dict={"imgs":sample_rate, "labels":sample_rate},
        clip_seg_num_dict={"imgs":sliding_window, "labels":sliding_window},
        sliding_window_dict={"imgs":sliding_window, "labels":sliding_window},
        sample_add_key_pair={"frames":"imgs"},
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "CompressedVideoStreamTransform",
        rgb = [
            dict(XToTensor = None),
            dict(ToFloat = None),
            dict(TensorPermute = dict(permute_list = [2, 0, 1])),
            dict(TensorImageResize = dict(size = (404, 720)))
        ],
        flow = [
            dict(XToTensor = None),
            dict(ToFloat = None),
            dict(TensorPermute = dict(permute_list = [2, 0, 1])),
            dict(TensorImageResize = dict(size = (404, 720)))
        ],
        res = [
            dict(XToTensor = None),
            dict(ToFloat = None),
            dict(TensorPermute = dict(permute_list = [2, 0, 1])),
            dict(TensorImageResize = dict(size = (404, 720)))
        ]
    )
)
