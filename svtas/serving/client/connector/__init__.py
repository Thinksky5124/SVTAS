'''
Author       : Thyssen Wen
Date         : 2023-10-30 14:47:40
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 14:55:34
Description  : file content
FilePath     : /SVTAS/svtas/serving/client/connector/__init__.py
'''
from .base_connector import BaseClientConnector
from .tritron_connector import TritronConnector

__all__ = [
    "BaseClientConnector", "TritronConnector"
]