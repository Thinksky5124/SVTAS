'''
Author       : Thyssen Wen
Date         : 2023-10-09 09:21:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 09:36:54
Description  : file content
FilePath     : /SVTAS/svtas/model/losses/base_loss.py
'''
from typing import Dict
from svtas.model_pipline.wrapper import TorchModel

class BaseLoss(TorchModel):
    def __init__(self) -> None:
        super().__init__()
        
    def init_weights(self, init_cfg: Dict = ...):
        pass