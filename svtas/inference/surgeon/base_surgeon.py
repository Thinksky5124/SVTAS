'''
Author       : Thyssen Wen
Date         : 2023-10-23 21:07:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-23 21:07:56
Description  : file content
FilePath     : /SVTAS/svtas/inference/surgeon/base_surgeon.py
'''
import abc
from typing import Any, Dict

class BaseModelSurgeon(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass