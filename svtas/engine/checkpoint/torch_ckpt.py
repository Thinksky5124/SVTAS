'''
Author       : Thyssen Wen
Date         : 2023-09-25 17:07:24
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 21:03:40
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/torch_ckpt.py
'''
import os
from typing import Any, Dict
import torch
from .base_checkpoint import BaseCheckpointor
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine_component')
class TorchCheckpointor(BaseCheckpointor):
    def __init__(self,
                 save_path: str = None,
                 load_path: str = None,
                 map_location = None) -> None:
        super().__init__(save_path, load_path)
        self.map_loaction = map_location
        self.cnt = 0

    def init_ckpt(self, *args, **kwargs):
        pass
    
    def save(self, save_dict: Dict, path: str = None, file_name: str = None) -> bool:
        if path is None:
            path = self.save_path
        if file_name is None:
            file_name = str(self.cnt)
            self.cnt += 1
        torch.save(save_dict, os.path.join(path, file_name + ".pt"))

    def load(self, path: str = None) -> Dict:
        if path is None:
            path = self.load_path
        return torch.load(path, map_location=self.map_loaction)
    
    def shutdown(self) -> None:
        return super().shutdown()