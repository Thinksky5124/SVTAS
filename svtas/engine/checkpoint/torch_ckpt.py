'''
Author       : Thyssen Wen
Date         : 2023-09-25 17:07:24
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 17:13:24
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/torch_ckpt.py
'''
from typing import Any
import torch
from .base_checkpoint import BaseCheckpointor
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine_component')
class TorchCheckpointor(BaseCheckpointor):
    def __init__(self) -> None:
        super().__init__()

    def save(self, *args: Any, **kwds: Any) -> bool:
        return super().save(*args, **kwds)
    
    def load(self, *args: Any, **kwds: Any) -> Any:
        return super().load(*args, **kwds)