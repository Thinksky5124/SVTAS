'''
Author       : Thyssen Wen
Date         : 2023-10-09 12:34:34
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-15 20:06:13
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/deepspeed_checkpoint.py
'''
import os
from typing import Dict

from svtas.utils import is_deepspeed_available, AbstractBuildFactory
if is_deepspeed_available():
    from deepspeed import DeepSpeedEngine

from .base_checkpoint import BaseCheckpointor

@AbstractBuildFactory.register('engine_component')
class DeepSpeedCheckpointor(BaseCheckpointor):
    def __init__(self, save_path: str = None, load_path: str = None) -> None:
        super().__init__(save_path, load_path)
    
    def init_ckpt(self, ds_engine):
        self.ds_engine = ds_engine
    
    def save(self, save_dict: Dict, path: str = None, file_name: str = None) -> bool:
        assert hasattr(self, "ds_engine"), "You must call `set_deepspeed_engine` before save or load DeepSpeedCheckpointor!"
        if path is None:
            path = self.save_path
        if file_name is None:
            file_name = str(self.cnt)
            self.cnt += 1
        self.ds_engine.save_checkpoint(path, file_name, client_state=save_dict)
    
    def load(self, path: str = None) -> Dict:
        assert hasattr(self, "ds_engine"), "You must call `set_deepspeed_engine` before save or load DeepSpeedCheckpointor!"
        if path is None:
            path = self.load_path
        load_path, client_state = self.ds_engine.load_checkpoint(path)
        return client_state
    
    def shutdown(self) -> None:
        return super().shutdown()