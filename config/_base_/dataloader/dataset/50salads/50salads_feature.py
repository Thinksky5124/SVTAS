'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:24:35
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:32:22
Description  : 50salads dataset config
FilePath     : /SVTAS/config/_base_/dataset/50salads/50salads_feaeture.py
'''
DATASET = dict(
    temporal_clip_batch_size = 1,
    video_batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/50salads/splits/train.split1.bundle",
        feature_path = "./data/50salads/features",
        gt_path = "./data/50salads/groundTruth",
        actions_map_file_path = "./data/50salads/mapping.txt",
        dataset_type = "50salads"
    ),
    test = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/50salads/splits/test.split1.bundle",
        feature_path = "./data/50salads/features",
        gt_path = "./data/50salads/groundTruth",
        actions_map_file_path = "./data/50salads/mapping.txt",
        dataset_type = "50salads"
    )
)

METRIC = dict(
    TAS = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/50salads/mapping.txt",
    file_output = False,
    score_output = False),
)