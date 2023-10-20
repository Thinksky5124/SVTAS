'''
Author       : Thyssen Wen
Date         : 2022-09-24 14:59:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 10:20:02
Description  : Infer Engine Class
FilePath     : /SVTAS/svtas/engine/infer_engine.py
'''
from .standalone_engine import StandaloneEngine
from svtas.loader.dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory
from typing import Dict, List

@AbstractBuildFactory.register('engine')
class StandaloneInferEngine(StandaloneEngine):
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 logger_dict: Dict,
                 record: Dict,
                 metric: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 running_mode='infer') -> None:
        super().__init__(model_name, model_pipline, logger_dict, record,
                         metric, iter_method, checkpointor, running_mode)
