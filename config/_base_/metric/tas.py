'''
Author       : Thyssen Wen
Date         : 2022-12-13 10:47:38
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-13 10:49:07
Description  : file content
FilePath     : /SVTAS/config/_base_/metric/tas.py
'''
METRIC = dict(
    TAS = dict(
        name = "TASegmentationMetric",
        overlap = [.1, .25, .5],
        actions_map_file_path = "./data/gtea/mapping.txt",
        file_output = False,
        score_output = False)
)