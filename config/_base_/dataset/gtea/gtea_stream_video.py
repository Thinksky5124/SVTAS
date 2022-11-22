'''
Author       : Thyssen Wen
Date         : 2022-10-27 18:30:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-22 18:01:30
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/gtea/gtea_stream_video.py
'''

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
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

# METRIC = dict(
#     TAS = dict(
#     name = "TASegmentationMetric",
#     overlap = [.1, .25, .5],
#     actions_map_file_path = "./data/gtea/mapping.txt",
#     file_output = False,
#     score_output = False),
#     ACC = dict(
#     name = "ConfusionMatrix",
#     actions_map_file_path = "./data/gtea/mapping.txt",
#     img_save_path = "./output",
#     need_plot = True,
#     need_color_bar = True,),
# )
METRIC = dict(
    TAS = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/gtea/mapping.txt",
    file_output = False,
    score_output = False),
)