'''
Author       : Thyssen Wen
Date         : 2022-10-27 18:25:10
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 21:34:36
Description  : gtea Dataset
FilePath     : /SVTAS/config/_base_/dataset/gtea/gtea_stream_feature.py
'''
DATASET = dict(
    temporal_clip_batch_size = 1,
    video_batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "FeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        feature_path = "./data/gtea/features",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        sliding_window = 1
    ),
    test = dict(
        name = "FeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        feature_path = "./data/gtea/features",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = 1
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