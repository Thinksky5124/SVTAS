'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:58:04
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 19:14:10
Description  : SVTAS module
FilePath     : /SVTAS/svtas/__init__.py
'''
__version__ = "0.2.0"
__author__ = "WujunWen"

from . import (engine, inference, loader, metric, model, model_pipline,
               optimizer, serving, api, utils, autotunning, compression,
               profiling, nas)
