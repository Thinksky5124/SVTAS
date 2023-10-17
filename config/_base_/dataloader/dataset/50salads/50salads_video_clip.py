'''
Author       : Thyssen Wen
Date         : 2023-02-22 10:11:08
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-22 10:11:12
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/50salads/50salads_video_clip.py
'''
DATASET = dict(
    
    batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "RawFrameClipSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/50salads/splits/train.split1.bundle",
        videos_path = "./data/50salads/Videos",
        gt_path = "./data/50salads/groundTruth",
        actions_map_file_path = "./data/50salads/mapping.txt",
        dataset_type = "50salads",
        sliding_window = 64
    ),
    test = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/50salads/splits/test.split1.bundle",
        videos_path = "./data/50salads/Videos",
        gt_path = "./data/50salads/groundTruth",
        actions_map_file_path = "./data/50salads/mapping.txt",
        dataset_type = "50salads",
        train_mode = False,
        sliding_window = 64
    )
)

METRIC = dict(
    TAS = dict(
        name = "TASegmentationMetric",
        overlap = [.1, .25, .5],
        actions_map_file_path = "./data/50salads/mapping.txt",
        file_output = False,
        score_output = False)
)