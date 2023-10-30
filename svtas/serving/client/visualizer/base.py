'''
Author       : Thyssen Wen
Date         : 2023-10-30 15:22:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 15:28:54
Description  : file content
FilePath     : /SVTAS/svtas/serving/client/visualizer/base.py
'''
import abc
from typing import Dict, Any
from svtas.utils.logger import BaseLogger, get_root_logger_instance

class BaseClientViusalizer(metaclass=abc.ABCMeta):
    logger: BaseLogger
    def __init__(self) -> None:
        self.logger = get_root_logger_instance()
    
    @abc.abstractmethod
    def show(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass