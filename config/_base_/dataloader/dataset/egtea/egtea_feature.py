'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:26:24
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-23 11:57:01
Description  : EGTEA dataset config
FilePath     : /SVTAS/config/_base_/dataset/egtea/egtea_feature.py
'''
DATASET = dict(
    
    batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/egtea/splits/train.split1.bundle",
        feature_path = "./data/egtea/features",
        gt_path = "./data/egtea/groundTruth",
        actions_map_file_path = "./data/egtea/mapping.txt",
        dataset_type = "egtea"
    ),
    test = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/egtea/splits/test.split1.bundle",
        feature_path = "./data/egtea/features",
        gt_path = "./data/egtea/groundTruth",
        actions_map_file_path = "./data/egtea/mapping.txt",
        dataset_type = "egtea"
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