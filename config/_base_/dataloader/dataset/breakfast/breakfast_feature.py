'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:26:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:41:19
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/breakfast/breakfast_feature.py
'''
DATASET = dict(
    
    batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/train.split1.bundle",
        feature_path = "./data/breakfast/features",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast"
    ),
    test = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/test.split1.bundle",
        feature_path = "./data/breakfast/features",
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