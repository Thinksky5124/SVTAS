'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:10:26
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:36:45
Description  : LiteFlowNet Config
FilePath     : /SVTAS/config/extract/extract_flow/liteflow_gtea.py
'''
MODEL = dict(
    architecture = "OpticalFlowEstimation",
    model = dict(
        name = "LiteFlowNetV3",
        pretrained = "./data/network-sintel.pytorch",
        freeze = True
    )
)

DATASET = dict(
    video_batch_size = 1,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = 64
    )
)

TRANSFORM = [
    dict(PILToTensor = None),
    dict(ToFloat = None),
    dict(NormalizeColorTo1 = None)
]