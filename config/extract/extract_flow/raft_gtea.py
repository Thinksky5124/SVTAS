'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:09:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-16 14:25:44
Description  : RAFT extract flow Config
FilePath     : /SVTAS/config/extract/extract_flow/raft_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py', '../../_base_/models/optical_flow_estimate/raft.py',
    '../../_base_/dataset/gtea/gtea_stream_video.py'
]
sliding_window = 32
clip_seg_num = 32
sample_rate = 1

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
        sliding_window = clip_seg_num
    )
)

POSTPRECESSING = dict(
    name = "OpticalFlowPostProcessing",
    fps = 15,
    need_visualize = False,
    sliding_window = sliding_window
)

PIPELINE = dict(
    name = "BaseDatasetPipline",
    decode = dict(
        name = "VideoDecoder",
        backend = dict(name='DecordContainer')
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
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ])
    )
)
