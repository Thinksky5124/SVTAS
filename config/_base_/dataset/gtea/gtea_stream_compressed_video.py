'''
Author       : Thyssen Wen
Date         : 2022-11-09 16:16:34
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-09 16:16:44
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/gtea/gtea_stream_compressed_video.py
'''
DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 2,
    num_workers = 2 * 2,
    train = dict(
        name = "CompressedVideoStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        sliding_window = 64,
        need_residual = True,
        need_mvs = True
    ),
    test = dict(
        name = "CompressedVideoStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = 64,
        need_residual = True,
        need_mvs = True
    )
)

METRIC = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/gtea/mapping.txt",
    file_output = False,
    score_output = False
)