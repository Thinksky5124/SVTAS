'''
Author       : Thyssen Wen
Date         : 2023-09-25 17:05:46
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 17:10:15
Description  : file content
FilePath     : /SVTAS/svtas/engine/checkpoint/__init_.py
'''
from .base_checkpoint import BaseCheckpointor
from .torch_ckpt import TorchCheckpointor

__all__ = [
    'BaseCheckpointor', 'TorchCheckpointor'
]