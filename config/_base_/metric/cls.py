'''
Author       : Thyssen Wen
Date         : 2022-12-13 10:48:02
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-13 10:48:21
Description  : file content
FilePath     : /SVTAS/config/_base_/metric/cls.py
'''
METRIC = dict(
    ACC = dict(
        name = "ConfusionMatrix",
        actions_map_file_path = "./data/gtea/mapping.txt",
        img_save_path = "./output",
        need_plot = False,
        need_color_bar = False,)
)