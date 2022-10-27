'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:10:26
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 11:24:02
Description  : LiteFlowNet Config
FilePath     : /SVTAS/config/extract_flow/liteflow_gtea.py
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
    video_path = "./data/gtea/Videos",
    file_list = "./data/gtea/splits/all_files.txt",
    dataset_type = "gtea",
    num_segments = 32,
    fps = 15
)

TRANSFORM = [
    dict(PILToTensor = None),
    dict(ToFloat = None),
    dict(NormalizeColorTo1 = None)
]