'''
Author       : Thyssen Wen
Date         : 2022-10-27 18:30:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-22 19:08:02
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/gtea/gtea_video_clip.py
'''

DATASET = dict(
    
    batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "RawFrameClipSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        sliding_window = 64
    ),
    test = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = 64
    )
)

METRIC = dict(
    TAS = dict(
        name = "TASegmentationMetric",
        overlap = [.1, .25, .5],
        actions_map_file_path = "./data/gtea/mapping.txt",
        file_output = False,
        score_output = False)
)