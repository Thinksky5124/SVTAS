'''
Author       : Thyssen Wen
Date         : 2023-09-25 17:07:24
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-04 17:32:32
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/torch_ckpt.py
'''
from typing import Any, Dict
import torch
from .base_checkpoint import BaseCheckpointor
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine_component')
class TorchCheckpointor(BaseCheckpointor):
    def __init__(self,
                 save_path: str = None,
                 load_path: str = None,
                 map_location=torch.device('cpu')) -> None:
        super().__init__(save_path, load_path)
        self.map_loaction = map_location

    def save(self, save_dict: Dict, path: str = None) -> bool:
        if path is None:
            path = self.save_path
        torch.save(save_dict, path)

    def load(self, path: str) -> Dict:
        if path is None:
            path = self.load_path
        return torch.load(path, map_location=self.map_loaction)