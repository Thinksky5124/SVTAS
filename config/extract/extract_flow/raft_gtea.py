'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:09:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 19:25:09
Description  : RAFT extract flow Config
FilePath     : /SVTAS/config/extract/extract_flow/raft_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py', '../../_base_/models/optical_flow_estimate/raft.py',
    '../../_base_/dataset/gtea_video.py'
]

DATASET = dict(
    video_path = "./data/gtea/Videos",
    file_list = "./data/gtea/splits/all_files.txt",
    dataset_type = "gtea",
    sliding_window = 32
)

TRANSFORM = [
    dict(PILToTensor = None),
    dict(ToFloat = None),
    dict(Normalize = dict(
        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
    ))
]