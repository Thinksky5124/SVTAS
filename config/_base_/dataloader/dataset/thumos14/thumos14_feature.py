'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:26:43
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:40:01
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/thumos14/thumos14_feature.py
'''
DATASET = dict(
    
    batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/thumos14/val_list.txt",
        feature_path = "./data/thumos14/features",
        gt_path = "./data/thumos14/groundTruth",
        actions_map_file_path = "./data/thumos14/mapping.txt",
        dataset_type = "thumos14"
    ),
    test = dict(
        name = "FeatureSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/thumos14/test_list.txt",
        feature_path = "./data/thumos14/features",
        gt_path = "./data/thumos14/groundTruth",
        actions_map_file_path = "./data/thumos14/mapping.txt",
        dataset_type = "thumos14"
    )
)

METRIC = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/thumos14/mapping.txt",
    file_output = False,
    score_output = False
)