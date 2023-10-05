'''
Author       : Thyssen Wen
Date         : 2023-09-25 17:06:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 20:39:36
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/base_checkpoint.py
'''
import os
import abc
from typing import Any, Dict

class BaseCheckpointor(metaclass=abc.ABCMeta):
    save_path: str
    load_path: str
    def __init__(self,
                 save_path: str = None,
                 load_path: str = None) -> None:
        if save_path is None:
            save_path = os.path.join(os.environ['SVTAS_LOG_DIR'], "ckpt")
        self.save_path = save_path
        self.load_path = load_path
    
    @property
    def load_flag(self) -> bool:
        if self.load_path is not None:
            return True
        return False
    
    @property
    def save_flag(self) -> bool:
        if self.save_path is not None:
            return True
        return False

    @abc.abstractmethod
    def save(self,
             save_dict: Dict,
             path: str = None) -> bool:
        raise NotImplementedError("You must implement save function!")
    
    @abc.abstractmethod
    def load(self, path: str = None) -> Dict:
        raise NotImplementedError("You must implement load function!")
