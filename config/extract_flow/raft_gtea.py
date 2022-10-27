'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:09:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 11:22:37
Description  : RAFT extract flow Config
FilePath     : /SVTAS/config/extract_flow/raft_gtea.py
'''
MODEL = dict(
    architecture = "OpticalFlowEstimation",
    model = dict(
        name = "RAFT",
        pretrained = "./data/raft-sintel.pth",
        extract_mode = True,
        freeze = True,
        mode = "sintel"
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