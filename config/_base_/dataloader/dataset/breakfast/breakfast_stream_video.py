'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:25:37
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 21:35:21
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/breakfast/breakfast_stream_video.py
'''
DATASET = dict(
    
    batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/train.split1.bundle",
        videos_path = "./data/breakfast/Videos",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = True,
        sliding_window = 64
    ),
    test = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/test.split1.bundle",
        videos_path = "./data/breakfast/Videos",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = False,
        sliding_window = 64
    )
)

METRIC = dict(
    TAS = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/breakfast/mapping.txt",
    file_output = False,
    score_output = False),
)