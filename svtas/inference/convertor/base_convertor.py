'''
Author       : Thyssen Wen
Date         : 2023-10-19 16:39:57
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 18:47:28
Description  : file content
FilePath     : /SVTAS/svtas/inference/convertor/base_convertor.py
'''
import abc
from typing import Any, Dict
from svtas.utils import get_log_root_path

class BaseModelConvertor(metaclass=abc.ABCMeta):
    def __init__(self,
                 export_path: str = None) -> None:
        if export_path is None:
            self.export_path = get_log_root_path()
        else:
            self.export_path = export_path
    
    @abc.abstractmethod
    def export(self, model: Any, data: Dict[str, Any], file_name: str, export_path: str = None):
        pass
    
    @abc.abstractmethod
    def init_convertor(self):
        pass
    
    @abc.abstractmethod
    def shutdown(self):
        pass