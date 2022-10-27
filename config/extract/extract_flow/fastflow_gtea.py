'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:10:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 11:21:01
Description  : FastFlow Config
FilePath     : /SVTAS/config/extract_flow/fastflow_gtea.py
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
    video_path = "./data/gtea/Videos",
    file_list = "./data/gtea/splits/all_files.txt",
    dataset_type = "gtea",
    num_segments = 32,
    fps = 15
)

TRANSFORM = [
    dict(PILToTensor = None),
    dict(ToFloat = None),
    dict(Normalize = dict(
        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
    ))
]