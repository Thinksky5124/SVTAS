'''
Author       : Thyssen Wen
Date         : 2022-12-13 10:47:44
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-13 10:48:56
Description  : file content
FilePath     : /SVTAS/config/_base_/metric/tap.py
'''
METRIC = dict(
    TAP = dict(
        name = "TAProposalMetric",
        actions_map_file_path = "./data/gtea/mapping.txt",
        max_proposal=100,)
)