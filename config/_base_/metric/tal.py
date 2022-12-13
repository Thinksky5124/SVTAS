'''
Author       : Thyssen Wen
Date         : 2022-12-13 10:47:51
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-13 10:48:47
Description  : file content
FilePath     : /SVTAS/config/_base_/metric/tal.py
'''
METRIC = dict(
    TAL = dict(
        name = "TALocalizationMetric",
        actions_map_file_path = "./data/gtea/mapping.txt",
        show_ovberlaps=[0.5, 0.75],)
)