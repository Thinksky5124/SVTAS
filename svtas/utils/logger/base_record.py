'''
Author       : Thyssen Wen
Date         : 2023-09-25 19:51:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 13:09:55
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/base_record.py
'''
import abc
from typing import List, Dict, Any

class BaseRecord(metaclass=abc.ABCMeta):
    def __init__(self,
                 addition_record: List[Dict] = None) -> None:
        self.addition_record = addition_record
        self._record = {}
    
    def __getitem__(self, key):
        return self._record[key]
    
    def __setitem__(self, key, value):
        self._record[key] = value
    
    def __iter__(self):
        return iter(self._record)

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

    def remove_one_record(self, name):
        self._record.pop(name)
    
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
    def accumulate_record(self):
        pass

    def save(self) -> Dict:
        return self._record

    def load(self, load_dict: Dict):
        self._record = load_dict