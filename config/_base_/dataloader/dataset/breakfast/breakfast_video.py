'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:25:37
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 09:50:15
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/breakfast/breakfast_video.py
'''
DATASET = dict(
    
    batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "RawFrameSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/train.split1.bundle",
        videos_path = "./data/breakfast/Videos",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast"
    ),
    test = dict(
        name = "RawFrameSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/test.split1.bundle",
        videos_path = "./data/breakfast/Videos",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast"
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