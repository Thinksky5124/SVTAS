'''
Author       : Thyssen Wen
Date         : 2022-11-21 10:59:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 10:59:27
Description  : file content
FilePath     : /SVTAS/config/_base_/schedules/grad_clip.py
'''
GRADCLIP = dict(
    name = "GradClip",
    max_norm=40,
    norm_type=2
)