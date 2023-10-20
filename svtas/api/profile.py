'''
Author: Thyssen Wen
Date: 2022-03-17 12:12:57
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:47:37
Description: test script api
FilePath     : /SVTAS/svtas/tasks/profile.py
'''
import torch
import torch.profiler
from ..utils.logger import get_logger
import time
import numpy as np
import datetime

from svtas.engine import BaseEngine
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import BaseModelPipline
from ..utils.collect_env import collect_env

@torch.no_grad()
def profile(local_rank,
            nprocs,
            cfg,
            args):
    logger = get_logger("SVTAS")
    model_name = cfg.model_name
    
    # env info logger
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # 1. Construct model.
    model_pipline: BaseModelPipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)
    device = model_pipline.device

    # 2. Construct dataset and dataloader.
    test_dataloader = AbstractBuildFactory.create_factory('dataloader').create(cfg.DATALOADER)
    
     # 3. build engine
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name
    profile_engine: BaseEngine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    profile_engine.set_dataloader(test_dataloader)
    profile_engine.running_mode = 'profile'
    
    # 6. resume engine
    if cfg.ENGINE.checkpointor.get('load_path', None) is not None:
        profile_engine.resume()
        
    # 7. run engine
    profile_engine.init_engine()
    profile_engine.run()
    profile_engine.shutdown()