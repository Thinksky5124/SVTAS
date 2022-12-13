'''
Author       : Thyssen Wen
Date         : 2022-12-13 10:47:57
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-13 10:48:36
Description  : file content
FilePath     : /SVTAS/config/_base_/metric/svtas.py
'''
METRIC = dict(
    SVTAS = dict(
        name = "SVTASegmentationMetric",
        overlap = [.1, .25, .5],
        segment_windows_size = 64,
        actions_map_file_path = "./data/gtea/mapping.txt",
        file_output = False,
        score_output = False),
)