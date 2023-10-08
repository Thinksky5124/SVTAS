'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:26:52
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 21:35:14
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/thumos14/thumos14_stream_video.py
'''
DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/thumos14/val_list.txt",
        videos_path = "./data/thumos14/Videos",
        gt_path = "./data/thumos14/groundTruth",
        actions_map_file_path = "./data/thumos14/mapping.txt",
        dataset_type = "thumos14",
        train_mode = True,
        sliding_window = 64
    ),
    test = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/thumos14/test_list.txt",
        videos_path = "./data/thumos14/Videos",
        gt_path = "./data/thumos14/groundTruth",
        actions_map_file_path = "./data/thumos14/mapping.txt",
        dataset_type = "thumos14",
        train_mode = False,
        sliding_window = 64
    )
)

METRIC = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/thumos14/mapping.txt",
    file_output = False,
    score_output = False
)