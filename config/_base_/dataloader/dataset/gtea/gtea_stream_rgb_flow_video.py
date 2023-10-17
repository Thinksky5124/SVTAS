'''
Author       : Thyssen Wen
Date         : 2022-11-09 16:15:34
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-09 16:15:36
Description  : file content
FilePath     : /SVTAS/config/_base_/dataset/gtea/gtea_stream_rgb_flow_video.py
'''
DATASET = dict(
    
    batch_size = 2,
    num_workers = 2 * 2,
    train = dict(
        name = "RGBFlowFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        videos_path = "./data/gtea/Videos",
        flows_path = './data/gtea/flow',
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        sliding_window = 64,
    ),
    test = dict(
        name = "RGBFlowFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        videos_path = "./data/gtea/Videos",
        flows_path = './data/gtea/flow',
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = 64,
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