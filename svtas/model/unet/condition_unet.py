'''
Author       : Thyssen Wen
Date         : 2023-10-12 15:38:43
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 14:43:52
Description  : file content
FilePath     : /SVTAS/svtas/model/unet/condition_unet.py
'''
import torch
import torch.nn as nn

class ConditionUnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def init_weights(self, init_cfg: dict = {}):
        pass
    
    def _clear_memory_buffer(self):
       pass