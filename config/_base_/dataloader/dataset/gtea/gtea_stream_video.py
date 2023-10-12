'''
Author       : Thyssen Wen
Date         : 2022-10-27 18:30:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-12 10:48:25
Description  : file content
FilePath     : /SVTAS/config/_base_/dataloader/dataset/gtea/gtea_stream_video.py
'''

DATASET = dict(
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

METRIC = dict(
    TAS = dict(
        name = "TASegmentationMetric",
        overlap = [.1, .25, .5],
        actions_map_file_path = "./data/gtea/mapping.txt",
        file_output = False,
        score_output = False),
    # TAP = dict(
    #     name = "TAProposalMetric",
    #     actions_map_file_path = "./data/gtea/mapping.txt",
    #     max_proposal=100,),
    # TAL = dict(
    #     name = "TALocalizationMetric",
    #     actions_map_file_path = "./data/gtea/mapping.txt",
    #     show_ovberlaps=[0.5, 0.75],),
)