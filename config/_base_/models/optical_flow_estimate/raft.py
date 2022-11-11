'''
Author       : Thyssen Wen
Date         : 2022-10-27 19:18:34
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-11 12:19:35
Description  : RAFT model
FilePath     : /SVTAS/config/_base_/models/optical_flow_estimate/raft.py
'''
MODEL = dict(
    architecture = "OpticalFlowEstimation",
    model = dict(
        name = "RAFT",
        pretrained = "./data/checkpoint/raft-sintel.pth",
        extract_mode = True,
        freeze = True,
        mode = "sintel"
    )
)