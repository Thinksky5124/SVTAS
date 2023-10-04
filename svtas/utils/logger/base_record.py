'''
Author       : Thyssen Wen
Date         : 2023-09-25 19:51:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-30 10:05:09
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/base_record.py
'''
import abc
from typing import List, Dict, Any

class BaseRecord(metaclass=abc.ABCMeta):
    def __init__(self,
                 mode,
                 addition_record: List[Dict] = None) -> None:
        self.mode = mode
        self.addition_record = addition_record
        self._record = {}

    @property
    def record_dict(self) -> Dict[str, Any]:
        return self._record
    
    @property
    def recodor_keys(self) -> List[str]:
        return list(self._record.keys())

    @abc.abstractmethod
    def add_record(self, name, fmt='f'):
        pass
    
    @abc.abstractmethod
    def update_one_record(self, name, value, n = 1):
        pass
    
    @abc.abstractmethod
    def get_one_record(self, name):
        pass

    @abc.abstractmethod
    def init_record(self):
        pass
    
    @abc.abstractmethod
    def update_record(self, update_dict: Dict):
        pass
    
    @abc.abstractmethod
    def update_loss_dict(self, update_dict: Dict):
        pass

    @abc.abstractmethod
    def accumulate_record(self):
        pass