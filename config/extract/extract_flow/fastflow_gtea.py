'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:10:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 14:36:48
Description  : FastFlow Config
FilePath     : /SVTAS/config/extract/extract_flow/fastflow_gtea.py
'''
MODEL = dict(
    architecture = "OpticalFlowEstimation",
    model = dict(
        name = "FastFlowNet",
        pretrained = "./data/fastflownet_ft_mix.pth",
        extract_mode = True,
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
    dict(Normalize = dict(
        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
    ))
]