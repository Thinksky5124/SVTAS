'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:26:33
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 21:35:19
Description  : EGTEA dataset Config
FilePath     : /SVTAS/config/_base_/dataset/egtea/egtea_stream_video.py
'''
DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/egtea/splits/train_split1.txt",
        videos_path = "./data/egtea/Videos",
        gt_path = "./data/egtea/groundTruth",
        actions_map_file_path = "./data/egtea/mapping.txt",
        dataset_type = "egtea",
        train_mode = True,
        sliding_window = 64
    ),
    test = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/egtea/splits/test_split1.txt",
        videos_path = "./data/egtea/Videos",
        gt_path = "./data/egtea/groundTruth",
        actions_map_file_path = "./data/egtea/mapping.txt",
        dataset_type = "egtea",
        train_mode = False,
        sliding_window = 64
    )
)

METRIC = dict(
    TAS = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/egtea/mapping.txt",
    file_output = False,
    score_output = False),
)