'''
Author       : Thyssen Wen
Date         : 2023-10-28 20:44:02
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-28 20:51:00
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/pickle_ckpt.py
'''
import os
from typing import Any, Dict
import pickle
from .base_checkpoint import BaseCheckpointor
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine_component')
class PickleCheckpointor(BaseCheckpointor):
    def __init__(self,
                 save_path: str = None,
                 load_path: str = None) -> None:
        super().__init__(save_path, load_path)
        self.cnt = 0
    
    def init_ckpt(self, *args, **kwargs):
        pass

    def save(self, save_dict: Dict, path: str = None, file_name: str = None) -> bool:
        if path is None:
            path = self.save_path
        if file_name is None:
            file_name = str(self.cnt)
            self.cnt += 1
        fil = open(os.path.join(path, file_name + ".pkl"), 'wb')
        pickle.dump(save_dict, fil)
    
    def load(self, path: str = None) -> Dict:
        if path is None:
            path = self.load_path
        return pickle.load(path)
    
    def shutdown(self) -> None:
        return super().shutdown()